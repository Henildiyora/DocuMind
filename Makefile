# DocuMind (Go) build targets.
#
# Prerequisites: Go 1.24+ and a C compiler (cgo), required by the tree-sitter
# grammars used for code-aware chunking.

BINARY := bin/documind
PKG := ./...

# Version stamped into the binary. Uses the nearest git tag when available,
# otherwise "dev". Release CI overrides this with the pushed tag.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -X github.com/Henildiyora/DocuMind/internal/cli.Version=$(VERSION)

# Install location for `make install` (user-writable by default = no sudo).
PREFIX ?= $(HOME)/.local

.PHONY: build test lint tidy run clean install

build:
	CGO_ENABLED=1 go build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/documind

# install a from-source build onto PATH. Override target dir with PREFIX, e.g.
#   sudo make install PREFIX=/usr/local
install: build
	install -d "$(PREFIX)/bin"
	install -m 0755 $(BINARY) "$(PREFIX)/bin/documind"
	@echo "Installed to $(PREFIX)/bin/documind"
	@case ":$$PATH:" in *":$(PREFIX)/bin:"*) ;; *) echo "Note: add $(PREFIX)/bin to your PATH.";; esac

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
