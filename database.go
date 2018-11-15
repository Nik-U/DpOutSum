package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type Distance uint32

type Database struct {
	Employers []string
	JobTitles []string
	Years     []uint16

	Employees []*Employee
}

type Employee struct {
	Id uint64

	// View filtering attributes
	Employer uint
	JobTitle uint
	Year     uint

	// Distance attributes
	Salary uint32
}

func (e *Employee) Distance(other *Employee) Distance {
	if e.Salary < other.Salary {
		return Distance(other.Salary - e.Salary)
	} else {
		return Distance(e.Salary - other.Salary)
	}
}

func ReadDatabase(r io.Reader, minPerEmployer uint, minPerJobTitle uint) (*Database, error) {
	db := &Database{}

	in := csv.NewReader(r)

	// Column indices in the CSV for the fields of interest
	idCol := -1
	employerCol := -1
	jobTitleCol := -1
	salaryCol := -1
	yearCol := -1

	// Tracking for unique values that can be used as selection filters
	employerSet := make(map[string]uint) // Maps value -> initial array index
	jobTitleSet := make(map[string]uint)
	yearSet := make(map[uint16]uint)

	// Counters for members of classes (for initial filtering)
	employerEmployees := make([]uint, 0) // Maps array index -> number of records (for filtering)
	jobTitleEmployees := make([]uint, 0)

	// Figure out where the columns are
	header, err := in.Read()
	if err != nil {
		return nil, errors.New(fmt.Sprintf("failed to read CSV header: %s", err))
	}
	for columnNum, columnName := range header {
		switch columnName {
		case "":
			idCol = columnNum
		case "Employer":
			employerCol = columnNum
		case "Job Title":
			jobTitleCol = columnNum
		case "Salary Paid":
			salaryCol = columnNum
		case "Calendar Year":
			yearCol = columnNum
		}
		// If you update this switch, don't forget to update the completeness check below
	}
	if idCol < 0 || employerCol < 0 || jobTitleCol < 0 || salaryCol < 0 || yearCol < 0 {
		return nil, errors.New("some expected columns were missing from the CSV header")
	}

	// Assemble a complete slice of all employees in the file; we will do the initial filtering later
	unfilteredEmployees := make([]*Employee, 0)
	for {
		record, err := in.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.New(fmt.Sprintf("failed to read CSV record: %s", err))
		}

		// Extract attributes from the CSV row
		employer := record[employerCol]
		jobTitle := record[jobTitleCol]
		salaryStr := record[salaryCol]

		// Convert identifier
		id, err := strconv.ParseUint(record[idCol], 10, 64)
		if err != nil {
			return nil, errors.New(fmt.Sprintf("data error: invalid identifier \"%s\"", record[idCol]))
		}

		// Track employer counts
		employerNum, knownEmployer := employerSet[employer]
		if !knownEmployer {
			employerNum = uint(len(employerSet))
			employerSet[employer] = employerNum
			employerEmployees = append(employerEmployees, 0)
		}
		employerEmployees[employerNum]++

		// Track job title counts
		jobTitleNum, knownJobTitle := jobTitleSet[jobTitle]
		if !knownJobTitle {
			jobTitleNum = uint(len(jobTitleSet))
			jobTitleSet[jobTitle] = jobTitleNum
			jobTitleEmployees = append(jobTitleEmployees, 0)
		}
		jobTitleEmployees[jobTitleNum]++

		// Keep track of unique years
		year, err := strconv.ParseUint(record[yearCol], 10, 16)
		if err != nil {
			return nil, errors.New(fmt.Sprintf("data error: invalid calendar year \"%s\"", record[yearCol]))
		}
		yearNum, knownYear := yearSet[uint16(year)]
		if !knownYear {
			yearNum = uint(len(db.Years))
			yearSet[uint16(year)] = yearNum
			db.Years = append(db.Years, uint16(year))
		}

		// Clean up the salary attribute
		if salaryStr[0] == '$' {
			salaryStr = salaryStr[1:]
		}
		salaryStr = strings.Replace(salaryStr, ",", "", -1)
		if salaryDecimal := strings.IndexRune(salaryStr, '.'); salaryDecimal >= 0 {
			if salaryDecimal == 0 {
				salaryStr = "0"
			} else {
				salaryStr = salaryStr[:salaryDecimal]
			}
		}
		salary, err := strconv.ParseUint(salaryStr, 10, 32)
		if err != nil {
			return nil, errors.New(fmt.Sprintf("data error: invalid salary \"%s\"", record[salaryCol]))
		}

		// Add the record to the tentative list
		// We might initially filter out this record later
		// The unique value references are to the full array of attributes, which will themselves be filtered
		employee := &Employee{
			Id:       id,
			Employer: employerNum,
			JobTitle: jobTitleNum,
			Year:     yearNum,
			Salary:   uint32(salary),
		}
		unfilteredEmployees = append(unfilteredEmployees, employee)
	}

	// Initially filter the database to our subset of interest
	// First exclude all attributes that are too small
	employerNumPatches := make(map[uint]uint, len(employerSet))
	for employer, employerNum := range employerSet {
		if employerEmployees[employerNum] >= minPerEmployer {
			employerNumPatches[employerNum] = uint(len(employerNumPatches))
			db.Employers = append(db.Employers, employer)
		}
	}
	jobTitleNumPatches := make(map[uint]uint, len(jobTitleSet))
	for jobTitle, jobTitleNum := range jobTitleSet {
		if jobTitleEmployees[jobTitleNum] >= minPerJobTitle {
			jobTitleNumPatches[jobTitleNum] = uint(len(jobTitleNumPatches))
			db.JobTitles = append(db.JobTitles, jobTitle)
		}
	}
	// Patch the indices already in the records to reference the filtered attribute sets
	for _, employee := range unfilteredEmployees {
		newEmployerIndex, validEmployer := employerNumPatches[employee.Employer]
		newJobTitleIndex, validJobTitle := jobTitleNumPatches[employee.JobTitle]
		if validEmployer && validJobTitle {
			employee.Employer = newEmployerIndex
			employee.JobTitle = newJobTitleIndex
			db.Employees = append(db.Employees, employee)
		}
	}

	return db, nil
}

func (db *Database) Filter(validEmployers []bool, validJobTitles []bool, validYears []bool, out chan<- bool) {
	for _, employee := range db.Employees {
		valid := validEmployers[employee.Employer] &&
			validJobTitles[employee.JobTitle] &&
			validYears[employee.Year]
		out <- valid
	}
	close(out)
}
