// Command documind is the DocuMind CLI: local hybrid search and grounded Q&A.
package main

import (
	"os"

	"github.com/Henildiyora/DocuMind/internal/cli"
)

func main() {
	os.Exit(cli.Execute())
}
