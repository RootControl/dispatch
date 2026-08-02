// Separate module so the pgx driver never enters dispatch's go.mod. The parent
// module's ./... skips any directory holding its own go.mod, so `go test ./...`
// at the repo root stays hermetic and dependency-free.
module github.com/RootControl/dispatch/tiers/sqldb/integration

go 1.25.0

require (
	github.com/RootControl/dispatch v0.0.0
	github.com/jackc/pgx/v5 v5.10.0
)

require (
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/text v0.29.0 // indirect
)

replace github.com/RootControl/dispatch => ../../..
