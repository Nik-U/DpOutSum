package main

type Lof struct {
	Db        *Database
	Neighbors *Knn

	K         uint64
	Threshold float64
}

// Reused per-thread memory for cache efficiency
type LofCache struct {
	Neighborhoods              [][]uint64
	CoreDistance               []Distance
	LocalReachabilityDensities []float64
}

func NewLof(db *Database, neighbors *Knn, k uint64, threshold float64) *Lof {
	lof := &Lof{
		Db:        db,
		Neighbors: neighbors,
		K:         k,
		Threshold: threshold,
	}
	return lof
}

func (lof *Lof) NewThreadCache() *LofCache {
	cache := &LofCache{
		CoreDistance:               make([]Distance, len(lof.Db.Employees)),
		LocalReachabilityDensities: make([]float64, len(lof.Db.Employees)),
	}
	cache.Neighborhoods = make([][]uint64, len(lof.Db.Employees))
	for i := range cache.Neighborhoods {
		cache.Neighborhoods[i] = make([]uint64, lof.K)
	}
	return cache
}

type OutlierHandler func(*Employee, float64) bool

// FindOutliers calls outlierHandler for each detected outlier in a subset defined by an inclusion mask. When
// outlierHandler returns false, the procedure immediately returns. When cache is local to the calling thread, this
// function is thread safe.
func (lof *Lof) FindOutliers(cache *LofCache, im *InclusionMask, outlierHandler OutlierHandler) {
	lof.computeCoreDistances(cache, im)
	lof.computeLrds(cache, im)
	lof.computeLofs(cache, im, outlierHandler)
}

func (lof *Lof) computeCoreDistances(cache *LofCache, im *InclusionMask) {
	for i := range cache.CoreDistance {
		if !im.IsIncluded(uint64(i)) {
			continue
		}
		numNeighbors := lof.Neighbors.KNearest(im, uint64(i), cache.Neighborhoods[i])
		cache.Neighborhoods[i] = cache.Neighborhoods[i][:numNeighbors]
		furthest := cache.Neighborhoods[i][numNeighbors-1]
		cache.CoreDistance[i] = lof.Db.Employees[i].Distance(lof.Db.Employees[furthest])
	}
}

func (lof *Lof) computeLrds(cache *LofCache, im *InclusionMask) {
	for i := range cache.LocalReachabilityDensities {
		if !im.IsIncluded(uint64(i)) {
			continue
		}

		var sum float64
		for _, j := range cache.Neighborhoods[i] {
			reachabilityDistance := lof.Db.Employees[j].Distance(lof.Db.Employees[i])
			if reachabilityDistance < cache.CoreDistance[j] {
				reachabilityDistance = cache.CoreDistance[j]
			}
			sum += float64(reachabilityDistance)
		}
		cache.LocalReachabilityDensities[i] = float64(len(cache.Neighborhoods[i])) / sum
	}
}

func (lof *Lof) computeLofs(cache *LofCache, im *InclusionMask, outlierHandler OutlierHandler) {
	for i := range cache.LocalReachabilityDensities {
		if !im.IsIncluded(uint64(i)) {
			continue
		}

		var sum float64
		for _, j := range cache.Neighborhoods[i] {
			sum += cache.LocalReachabilityDensities[j]
		}
		score := sum / (float64(len(cache.Neighborhoods[i])) * cache.LocalReachabilityDensities[i])

		if score >= lof.Threshold {
			if !outlierHandler(lof.Db.Employees[i], score) {
				return
			}
		}
	}
}
