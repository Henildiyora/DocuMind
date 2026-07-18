# DocuMind (Go) build targets.
#
# Prerequisites: Go 1.24+ and a C compiler (cgo), required by the tree-sitter
# grammars used for code-aware chunking.

BINARY := bin/documind
PKG := ./...

.PHONY: build test lint tidy run clean

build:
	CGO_ENABLED=1 go build -o $(BINARY) ./cmd/documind

test:
	CGO_ENABLED=1 go test $(PKG)

lint:
	go vet $(PKG)
	gofmt -l .

tidy:
	go mod tidy

run: build
	./$(BINARY) $(ARGS)

clean:
	rm -rf bin
