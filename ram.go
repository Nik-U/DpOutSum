package main

import (
	"fmt"
	"log"
	"runtime"
)

func CleanRam(lg *log.Logger) {
	// Force garbage collection
	runtime.GC()
	lg.Println(RamStats())
}

func RamStats() string {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	return fmt.Sprintf("Heap: %.0f MiB; Sys: %.0f MiB; Mallocs: %d", float64(memStats.Alloc)/1024.0/1024.0, float64(memStats.Sys)/1024.0/1024.0, memStats.Mallocs)
}
