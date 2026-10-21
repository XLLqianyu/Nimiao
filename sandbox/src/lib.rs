//! Workspace-scoped path validation.
//!
//! Every filesystem tool funnels model-supplied paths through
//! [`Sandbox::validate_path`] before touching the disk. The target does not
//! have to exist: writes need to create new files and directories. Instead the
//! validator canonicalizes the deepest existing ancestor and checks that this
//! ancestor stays inside the workspace root. That catches both `..` traversal
//! and symlink escapes while still accepting not-yet-created paths.

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Component, Path, PathBuf};

#[derive(Debug)]
pub enum SandboxError {
    EscapesWorkspace { path: PathBuf, root: PathBuf },
    Io(io::Error),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandboxError::EscapesWorkspace { path, root } => write!(
                f,
                "path '{}' is outside the workspace '{}'",
                path.display(),
                root.display()
            ),
            SandboxError::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SandboxError::Io(err) => Some(err),
            SandboxError::EscapesWorkspace { .. } => None,
        }
    }
}

impl From<io::Error> for SandboxError {
    fn from(err: io::Error) -> Self {
        SandboxError::Io(err)
    }
}

/// A filesystem boundary rooted at one workspace directory.
#[derive(Debug, Clone)]
pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    /// Creates a sandbox rooted at an existing directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, SandboxError> {
        let canonical = root.as_ref().canonicalize()?;
        Ok(Self {
            root: strip_verbatim(&canonical),
        })
    }

    /// Creates a sandbox rooted at the current working directory.
    pub fn current() -> Result<Self, SandboxError> {
        Self::new(std::env::current_dir()?)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves `path` against the workspace root and verifies that it stays
    /// inside it.
    ///
    /// Returns an absolute path with symlinks in the existing prefix resolved.
    /// Trailing components that do not exist yet are appended unchanged, so the
    /// caller can create them afterwards.
    pub fn validate_path(&self, path: impl AsRef<Path>) -> Result<PathBuf, SandboxError> {
        let raw_input = path.as_ref();
        let escapes = || SandboxError::EscapesWorkspace {
            path: raw_input.to_path_buf(),
            root: self.root.clone(),
        };

        if is_device_path(raw_input) {
            return Err(escapes());
        }

        // Accept verbatim paths (`\\?\C:\...`) by normalizing them first; they
        // reach us when callers pass back a path produced by `canonicalize`.
        let raw = strip_verbatim(raw_input);
        let raw = raw.as_path();

        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else if raw.has_root() || has_prefix(raw) {
            // Root-relative (`\foo`) or drive-relative (`C:foo`) paths mean
            // different things depending on process state; refuse them.
            return Err(escapes());
        } else {
            self.root.join(raw)
        };
        let normalized = normalize_lexical(&joined);

        // Find the deepest ancestor that exists on disk. Whatever is below it
        // is a plain name and cannot redirect the path elsewhere.
        let mut ancestor = normalized.as_path();
        let mut missing: Vec<OsString> = Vec::new();
        loop {
            match std::fs::symlink_metadata(ancestor) {
                Ok(_) => break,
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    let name = ancestor.file_name().ok_or_else(escapes)?.to_os_string();
                    missing.push(name);
                    ancestor = ancestor.parent().ok_or_else(escapes)?;
                }
                Err(err) => return Err(SandboxError::Io(err)),
            }
        }

        let canonical = strip_verbatim(&ancestor.canonicalize()?);
        if !is_within(&self.root, &canonical) {
            return Err(escapes());
        }

        let mut resolved = canonical;
        for name in missing.into_iter().rev() {
            resolved.push(name);
        }
        Ok(resolved)
    }
}

/// Rebuilds a path without `.` or `..` components.
///
/// `..` pops a preceding normal component, but never pops past a root or a
/// Windows drive prefix.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let can_pop = matches!(out.components().next_back(), Some(Component::Normal(_)));
                if can_pop {
                    out.pop();
                }
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                out.push(component.as_os_str());
            }
        }
    }
    out
}

fn has_prefix(path: &Path) -> bool {
    matches!(path.components().next(), Some(Component::Prefix(_)))
}

/// Returns true when `candidate` is `root` itself or lives below it.
fn is_within(root: &Path, candidate: &Path) -> bool {
    let mut root_components = root.components();
    let mut candidate_components = candidate.components();
    loop {
        match (root_components.next(), candidate_components.next()) {
            (None, _) => return true,
            (Some(_), None) => return false,
            (Some(left), Some(right)) => {
                if !component_eq(&left, &right) {
                    return false;
                }
            }
        }
    }
}

#[cfg(windows)]
fn component_eq(left: &Component<'_>, right: &Component<'_>) -> bool {
    left.as_os_str().eq_ignore_ascii_case(right.as_os_str())
}

#[cfg(not(windows))]
fn component_eq(left: &Component<'_>, right: &Component<'_>) -> bool {
    left == right
}

#[cfg(windows)]
fn is_device_path(path: &Path) -> bool {
    let text = path.as_os_str().to_string_lossy().to_ascii_lowercase();
    // `\\.\` addresses devices such as `\\.\PhysicalDrive0`, not files.
    if text.starts_with("\\\\.\\") {
        return true;
    }
    // The global namespace reaches raw device objects and bypasses the
    // workspace entirely.
    if text.starts_with("\\\\?\\globalroot") {
        return true;
    }
    // Named pipes and mailslots are not files.
    text.starts_with("\\\\?\\pipe\\") || text.starts_with("\\\\?\\mailslot\\")
}

#[cfg(not(windows))]
fn is_device_path(_path: &Path) -> bool {
    false
}

/// Removes the Windows verbatim prefix so returned paths read naturally.
#[cfg(windows)]
fn strip_verbatim(path: &Path) -> PathBuf {
    let text = path.as_os_str().to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        if let Some(unc) = rest
            .get(..4)
            .filter(|prefix| prefix.eq_ignore_ascii_case("unc\\"))
        {
            return PathBuf::from(format!(r"\\{}", &rest[unc.len()..]));
        }
        return PathBuf::from(rest);
    }
    path.to_path_buf()
}

#[cfg(not(windows))]
fn strip_verbatim(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "nimiao-sandbox-{}-{}-{}",
            std::process::id(),
            seq,
            tag
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn resolves_relative_and_absolute_paths() {
        let root = temp_dir("basic");
        let sandbox = Sandbox::new(&root).unwrap();

        let relative = sandbox.validate_path("nested/file.txt").unwrap();
        assert!(relative.starts_with(sandbox.root()));
        assert!(relative.ends_with("nested/file.txt"));

        let absolute = sandbox
            .validate_path(root.join("nested").join("file.txt"))
            .unwrap();
        assert_eq!(relative, absolute);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_parent_traversal() {
        let root = temp_dir("traversal");
        let sandbox = Sandbox::new(&root).unwrap();

        assert!(sandbox.validate_path("../outside.txt").is_err());
        assert!(sandbox.validate_path("nested/../../outside.txt").is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_absolute_paths_outside_the_workspace() {
        let root = temp_dir("outside");
        let other = temp_dir("other");
        let sandbox = Sandbox::new(&root).unwrap();

        assert!(sandbox.validate_path(other.join("file.txt")).is_err());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&other).ok();
    }

    #[test]
    fn accepts_missing_paths_below_existing_ancestors() {
        let root = temp_dir("missing");
        let sandbox = Sandbox::new(&root).unwrap();

        let path = sandbox.validate_path("a/b/c/file.txt").unwrap();
        assert!(path.starts_with(sandbox.root()));
        assert!(path.ends_with("a/b/c/file.txt"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn normalizes_inner_dot_segments() {
        let root = temp_dir("dots");
        let sandbox = Sandbox::new(&root).unwrap();

        let path = sandbox.validate_path("a/./b/../c.txt").unwrap();
        assert!(path.ends_with("a/c.txt"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_dot_dot_that_climbs_above_the_root() {
        let root = temp_dir("climb");
        let sandbox = Sandbox::new(&root).unwrap();

        // Climbing above the workspace root must never resolve inside it.
        let result = sandbox.validate_path("../../../../../../etc/passwd");
        assert!(result.is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(windows)]
    #[test]
    fn rejects_device_paths() {
        let root = temp_dir("device");
        let sandbox = Sandbox::new(&root).unwrap();

        assert!(sandbox.validate_path(r"\\.\PhysicalDrive0").is_err());
        assert!(sandbox
            .validate_path(r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1")
            .is_err());
        assert!(sandbox.validate_path(r"C:\Windows\System32").is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_that_point_outside() {
        use std::os::unix::fs::symlink;

        let root = temp_dir("symlink");
        let other = temp_dir("symlink-target");
        symlink(&other, root.join("escape")).unwrap();

        let sandbox = Sandbox::new(&root).unwrap();
        assert!(sandbox.validate_path("escape/file.txt").is_err());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&other).ok();
    }

    #[cfg(windows)]
    #[test]
    fn rejects_symlinks_that_point_outside() {
        let root = temp_dir("symlink");
        let other = temp_dir("symlink-target");
        if std::os::windows::fs::symlink_dir(&other, root.join("escape")).is_err() {
            // Creating symlinks needs developer mode or elevation.
            std::fs::remove_dir_all(&root).ok();
            std::fs::remove_dir_all(&other).ok();
            return;
        }

        let sandbox = Sandbox::new(&root).unwrap();
        assert!(sandbox.validate_path("escape/file.txt").is_err());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&other).ok();
    }
}
