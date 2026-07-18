package indexer

import (
	"context"
	"path/filepath"
	"time"

	"github.com/gofrs/flock"
)

// lockPath returns the exclusive index lock path, kept NEXT TO .documind/ (not
// inside it) so reset/destroy never deletes an active lock inode. Same filename
// as the Python build.
func lockPath(projectRoot string) string {
	return filepath.Join(projectRoot, ".documind-index.lock")
}

// withIndexLock runs fn while holding an exclusive per-project index lock, so
// only one indexer touches a project at a time. Returns ErrLocked if the lock
// cannot be acquired within the timeout.
func withIndexLock(projectRoot string, timeout time.Duration, fn func() error) error {
	lock := flock.New(lockPath(projectRoot))
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	got, err := lock.TryLockContext(ctx, 100*time.Millisecond)
	if err != nil {
		return err
	}
	if !got {
		return ErrLocked
	}
	defer lock.Unlock()
	return fn()
}
