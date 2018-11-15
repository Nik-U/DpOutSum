package main

import (
	"log"
	"runtime"
	"sort"
	"time"
)

type Knn struct {
	Db               *Database
	NearestNeighbors [][]uint64 // Sorted lists of nearest neighbor indices in Db, computed once
}

// Inclusion mask is independently stored so it can be used per-thread
type InclusionMask struct {
	Mask  []uint8
	Count uint64
}

func NewInclusionMask(db *Database) *InclusionMask {
	return &InclusionMask{
		Mask:  make([]uint8, (len(db.Employees)+7)/8),
		Count: 0,
	}
}

func (im *InclusionMask) Fill(inclusion <-chan bool) {
	var b, bit uint
	for {
		include, more := <-inclusion
		if !more {
			break
		}

		if bit == 0 {
			im.Mask[b] = 0
		}

		if include {
			im.Mask[b] |= (1 << (7 - bit))
			im.Count++
		}
		bit++
		if bit > 7 {
			bit = 0
			b++
		}
	}
}

func (im *InclusionMask) IsIncluded(i uint64) bool {
	b := i / 8
	bit := i % 8
	return im.Mask[b]&(1<<(7-bit)) != 0
}

func NewKnn(db *Database, lg *log.Logger, printFrequency time.Duration) *Knn {
	knn := &Knn{Db: db}
	knn.precomputeDistances(lg, printFrequency)
	return knn
}

func (knn *Knn) precomputeDistances(lg *log.Logger, printFrequency time.Duration) {
	n := uint64(len(knn.Db.Employees))
	knn.NearestNeighbors = make([][]uint64, n)
	for i := range knn.NearestNeighbors {
		knn.NearestNeighbors[i] = make([]uint64, n-1) // We can't neighbor ourselves
	}

	// Compute the distance array in parallel because this is an O(n^2) operation

	workChan := make(chan uint64)
	finishedChan := make(chan struct{})
	workerCount := runtime.NumCPU()
	for worker := 0; worker < workerCount; worker++ {
		go func() {
			defer func() { finishedChan <- struct{}{} }()
			for {
				i, moreWork := <-workChan
				if !moreWork {
					break
				}

				distances := make([]struct {
					target   uint64
					distance Distance
				}, n)
				for j := uint64(0); j < n; j++ { // Distance may not be symmetrical
					distances[j].target = j
					if i == j {
						distances[j].distance = 0
					} else {
						distances[j].distance = knn.Db.Employees[i].Distance(knn.Db.Employees[j])
					}
				}
				sort.Slice(distances, func(a, b int) bool { return distances[a].distance < distances[b].distance })
				next := 0
				for _, neighbor := range distances {
					if neighbor.target == i {
						continue
					}
					knn.NearestNeighbors[i][next] = neighbor.target
					next++
				}
			}
		}()
	}

	lastPrint := time.Now()
	for i := uint64(0); i < n; i++ {
		workChan <- i
		if time.Since(lastPrint) >= printFrequency {
			lg.Printf("Computed %d / %d neighbor sets (%.2f%%). %s\n", i, n, float64(i)/float64(n)*100.0, RamStats())
			lastPrint = time.Now()
		}
	}
	close(workChan)
	for worker := 0; worker < workerCount; worker++ {
		<-finishedChan
	}
}

// KNearest finds the K nearest neighbors of i in O(n) worst-case time and O(1) best-case time. The inclusion mask
// specifies which records are considered part of the subset. "out" is filled with the closest neighbors. The function
// returns the actual number of neighbors found (it may be less than len(out)). This function is thread safe.
func (knn *Knn) KNearest(im *InclusionMask, i uint64, out []uint64) uint64 {
	var validNeighbors uint64
	for _, neighbor := range knn.NearestNeighbors[i] {
		if !im.IsIncluded(neighbor) {
			continue
		}
		out[validNeighbors] = neighbor
		validNeighbors++
		if validNeighbors >= uint64(len(out)) {
			break
		}
	}
	return validNeighbors
}
