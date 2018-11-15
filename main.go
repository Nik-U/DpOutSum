package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"time"
)

type Context struct {
	EmployersIncluded []bool
	JobTitlesIncluded []bool
	YearsIncluded     []bool
}

func NewContext(db *Database) *Context {
	return &Context{
		EmployersIncluded: make([]bool, len(db.Employers)),
		JobTitlesIncluded: make([]bool, len(db.JobTitles)),
		YearsIncluded:     make([]bool, len(db.Years)),
	}
}

func (ctx *Context) Copy(other *Context) {
	copy(ctx.EmployersIncluded, other.EmployersIncluded)
	copy(ctx.JobTitlesIncluded, other.JobTitlesIncluded)
	copy(ctx.YearsIncluded, other.YearsIncluded)
}

func (ctx *Context) WriteTo(w io.Writer) {
	dumpIncluded := func(arr []bool) {
		fmt.Fprintf(w, "[")
		first := true
		for i, included := range arr {
			if !included {
				continue
			}
			if first {
				first = false
			} else {
				fmt.Fprint(w, ", ")
			}
			fmt.Fprintf(w, "%d", i)
		}
		fmt.Fprintln(w, "]")
	}

	fmt.Fprintln(w, "{")
	fmt.Fprintf(w, "  Employers: ")
	dumpIncluded(ctx.EmployersIncluded)
	fmt.Fprintf(w, "  Job Titles: ")
	dumpIncluded(ctx.JobTitlesIncluded)
	fmt.Fprintf(w, "  Calendar Years: ")
	dumpIncluded(ctx.YearsIncluded)
	fmt.Fprintln(w, "}")
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s INFILE OUTFILE\n", os.Args[0])
		os.Exit(1)
	}

	lg := log.New(os.Stderr, "", log.Ldate|log.Ltime)

	inFile, err := os.Open(os.Args[1])
	if err != nil {
		lg.Fatalf("Failed to open input file for reading: %s\n", err)
	}
	defer inFile.Close()

	outRaw, err := os.Create(fmt.Sprintf("%s.gz", os.Args[2]))
	if err != nil {
		lg.Fatalf("Failed to open output file for writing: %s\n", err)
		os.Exit(1)
	}
	defer outRaw.Close()
	outFile := gzip.NewWriter(outRaw)
	defer outFile.Close()

	lg.Printf("Using parallelism over %d threads\n", runtime.NumCPU())

	lg.Printf("Reading records from input file \"%s\"\n", os.Args[1])

	const MinPerEmployer = 3000
	const MinPerJobTitle = 3000

	db, err := ReadDatabase(inFile, MinPerEmployer, MinPerJobTitle)
	if err != nil {
		lg.Fatalf("Failed to read employee database: %s\n", err)
		os.Exit(1)
	}

	lg.Printf("Database contains %d records (after initial filtering)\n", len(db.Employees))

	// The output file needs to have the list of attributes because they are scrambled with each load
	fmt.Fprintln(outFile, "Employers:")
	for i, employer := range db.Employers {
		fmt.Fprintf(outFile, "  %d: %s\n", i, employer)
	}
	fmt.Fprintln(outFile, "\nJob Titles:")
	for i, jobTitle := range db.JobTitles {
		fmt.Fprintf(outFile, "  %d: %s\n", i, jobTitle)
	}
	fmt.Fprintln(outFile, "\nCalendar Years:")
	for i, year := range db.Years {
		fmt.Fprintf(outFile, "  %d: %d\n", i, year)
	}
	fmt.Fprintln(outFile)

	// Precompute nearest neighbors
	const K = 20
	const OutlierThreshold = 1.5

	const PrintFrequency = time.Second * 30

	lg.Println("Precomputing nearest neighbors for all records")
	neighbors := NewKnn(db, lg, PrintFrequency)
	lg.Println("Completed nearest neighbor computation")

	lof := NewLof(db, neighbors, K, OutlierThreshold)

	CleanRam(lg)

	// Create an original context
	const OrigCtxEmployersCount = 6
	const OrigCtxJobTitlesCount = 5
	const OrigCtxYearsCount = 5
	ctx := NewContext(db)
	for i := 0; i < OrigCtxEmployersCount; i++ {
		ctx.EmployersIncluded[i] = true
	}
	for i := 0; i < OrigCtxJobTitlesCount; i++ {
		ctx.JobTitlesIncluded[i] = true
	}
	for i := 0; i < OrigCtxYearsCount; i++ {
		ctx.YearsIncluded[i] = true
	}
	lg.Printf("Formed original context with %d employers, %d job titles, and %d years\n", OrigCtxEmployersCount, OrigCtxJobTitlesCount, OrigCtxYearsCount)
	fmt.Fprintln(outFile, "Original context:")
	ctx.WriteTo(outFile)
	fmt.Fprintln(outFile)

	// Find the first outlier in this original context
	origIm := NewInclusionMask(db)
	origCache := lof.NewThreadCache()
	var origOutlier *Employee
	var origScore float64
	FindOutliers(db, lof, origCache, origIm, ctx, func(outlier *Employee, score float64) bool {
		origOutlier = outlier
		origScore = score
		return false
	})
	if origOutlier == nil {
		lg.Fatalln("Error: original context contains no outliers!")
	}
	lg.Printf("First outlier in original context is ID #%d with LOF %f\n", origOutlier.Id, origScore)
	fmt.Fprintf(outFile, "Original outlier with LOF %f: ID #%d, employer %d, job title %d, calendar year %d\n\n", origScore,
		origOutlier.Id, origOutlier.Employer, origOutlier.JobTitle, origOutlier.Year)

	// Now try all other possible superset contexts to see if this record is still an outlier
	// We do this in parallel for performance

	workerCount := runtime.NumCPU()

	var flippableVariables float64
	flippableVariables += float64(len(db.Employers) - OrigCtxEmployersCount)
	flippableVariables += float64(len(db.JobTitles) - OrigCtxJobTitlesCount)
	flippableVariables += float64(len(db.Years) - OrigCtxYearsCount)
	totalContexts := uint64(math.Exp2(flippableVariables))
	lg.Printf("Scanning %d contexts with %d threads\n", totalContexts, workerCount)
	scanStartTime := time.Now()

	type matchingOutlier struct {
		employee *Employee
		score    float64
	}
	type matchingContext struct {
		context       *Context
		popSize       uint64
		outlierList   []matchingOutlier
		printedNotice chan struct{}
	}

	workChan := make(chan *Context)
	ctxReuseChan := make(chan *Context, workerCount)
	matchingContextChan := make(chan *matchingContext)
	finishedChan := make(chan struct{})
	for worker := 0; worker < workerCount; worker++ {
		go func() {
			defer func() { finishedChan <- struct{}{} }()

			// Thread local storage that gets reused between contexts under analysis
			cache := lof.NewThreadCache()
			im := NewInclusionMask(db)
			match := &matchingContext{
				printedNotice: make(chan struct{}),
			}

			for {
				// Get a new context to analyze
				workCtx, more := <-workChan
				if !more {
					break
				}
				match.context = workCtx

				// Gather a list of all outliers in this sub-population
				match.outlierList = match.outlierList[:0]
				thisContextMatches := false
				FindOutliers(db, lof, cache, im, workCtx, func(employee *Employee, score float64) bool {
					foundMatch := employee == origOutlier
					if foundMatch {
						thisContextMatches = true
					}
					match.outlierList = append(match.outlierList, matchingOutlier{employee: employee, score: score})
					return true
				})
				match.popSize = im.Count // Filled by FindOutliers call

				// If the original outlier appears, send this context for printing
				if thisContextMatches {
					matchingContextChan <- match
					// Wait until it gets printed before processing more work
					// This prevents us from clobbering the outlier list buffer while it is being printed
					<-match.printedNotice
				}

				// Free up the context memory for reuse
				ctxReuseChan <- workCtx
			}
		}()
	}

	// Goroutine for logging results
	var foundContexts uint64
	go func() {
		defer func() { finishedChan <- struct{}{} }()
		for {
			match, more := <-matchingContextChan
			if !more {
				break
			}
			fmt.Fprintf(outFile, "Matching context with population size %d:\n", match.popSize)
			match.context.WriteTo(outFile)
			fmt.Fprintln(outFile, "Outliers in matching context:")
			for _, outlier := range match.outlierList {
				fmt.Fprintf(outFile, "  ID #%d with LOF %f\n", outlier.employee.Id, outlier.score)
			}
			fmt.Fprintln(outFile)
			foundContexts++

			// Wake up the worker once more
			match.printedNotice <- struct{}{}
		}
	}()

	// We maintain a pool of allocated work contexts
	// This allows workers to access the memory without being clobbered by the work assignment main thread
	// Reusing these contexts avoids the main source of mallocs in the critical path
	for i := 0; i < workerCount; i++ {
		ctxReuseChan <- NewContext(db)
	}
	CleanRam(lg)

	// Enumerate all possible supersets by flipping unused attribute values through all permutations
	lastPrint := time.Now()
	var processedContexts uint64
	originalContext := true
	RecursivePermute(ctx.EmployersIncluded[OrigCtxEmployersCount:], func() {
		RecursivePermute(ctx.JobTitlesIncluded[OrigCtxJobTitlesCount:], func() {
			RecursivePermute(ctx.YearsIncluded[OrigCtxYearsCount:], func() {
				// The first iteration is always unchanged from the start
				if originalContext {
					originalContext = false
					return
				}

				// Get an empty work context
				workCtx := <-ctxReuseChan

				// Dispatch the work context
				workCtx.Copy(ctx)
				workChan <- workCtx

				processedContexts++
				if time.Since(lastPrint) >= PrintFrequency {
					lg.Printf("Processed %d / %d contexts (%.2f%%). %s\n", processedContexts, totalContexts, float64(processedContexts)/float64(totalContexts)*100.0, RamStats())
					lastPrint = time.Now()
				}
			})
		})
	})
	close(workChan)
	for worker := 0; worker < workerCount; worker++ {
		<-finishedChan
	}
	close(matchingContextChan)
	<-finishedChan

	lg.Printf("Found %d matching contexts in %s\n", foundContexts, time.Since(scanStartTime))
}

func FindOutliers(db *Database, lof *Lof, cache *LofCache, im *InclusionMask, ctx *Context, outlierHandler OutlierHandler) {
	inclusion := make(chan bool)
	go db.Filter(ctx.EmployersIncluded, ctx.JobTitlesIncluded, ctx.YearsIncluded, inclusion)
	im.Fill(inclusion)

	const MinPopulationSize = 20
	if im.Count < MinPopulationSize {
		return
	}

	lof.FindOutliers(cache, im, outlierHandler)
}

func RecursivePermute(slice []bool, handler func()) {
	if len(slice) <= 0 {
		handler()
		return
	}
	slice[0] = false
	RecursivePermute(slice[1:], handler)
	slice[0] = true
	RecursivePermute(slice[1:], handler)
}
