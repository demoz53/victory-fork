//! What of the workspace's `Cargo.lock` and `Cargo.toml` goes into the
//! translator fingerprint (`build.rs`): only what can change what this
//! crate is built from. Shared by the build script and the crate's tests
//! (`src/lib.rs`, `fingerprint_inputs_tests`), so it has no dependencies.

/// The `[[package]]` blocks of `lock` (a `Cargo.lock`'s text) that `root`
/// depends on, itself included, at any depth: their text, each block's
/// lines as they are, the blocks sorted by name and version, after the
/// lock's `version` line. Another crate's lines (a dependency an unrelated
/// crate of the workspace takes) leave it as it is; a version or checksum
/// of a crate in the closure changes it.
pub fn lock_closure(lock: &str, root: &str) -> String {
    struct Package<'a> {
        name: &'a str,
        version: &'a str,
        deps: Vec<&'a str>,
        text: String,
    }
    let mut header = String::new();
    let mut packages: Vec<Package> = Vec::new();
    let mut current: Option<Package> = None;
    let mut in_deps = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            packages.extend(current.take());
            current = Some(Package { name: "", version: "", deps: Vec::new(), text: String::new() });
            in_deps = false;
            continue;
        }
        let Some(p) = current.as_mut() else {
            if line.starts_with("version") {
                header = line.to_string();
            }
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        p.text.push_str(line);
        p.text.push('\n');
        if in_deps {
            if line.trim() == "]" {
                in_deps = false;
            } else if let Some(dep) = line.trim().trim_end_matches(',').strip_prefix('"').and_then(|d| d.strip_suffix('"')) {
                p.deps.push(dep);
            }
        } else if let Some(v) = line.strip_prefix("name = ") {
            p.name = v.trim_matches('"');
        } else if let Some(v) = line.strip_prefix("version = ") {
            p.version = v.trim_matches('"');
        } else if line.starts_with("dependencies = [") {
            in_deps = !line.trim_end().ends_with(']');
        }
    }
    packages.extend(current);
    // A dependency is named "name" (the one version in the lock) or
    // "name version".
    let find = |dep: &str| -> Vec<usize> {
        let (name, version) = dep.split_once(' ').map_or((dep, None), |(n, v)| (n, Some(v)));
        (0..packages.len())
            .filter(|&i| packages[i].name == name && version.is_none_or(|v| packages[i].version == v))
            .collect()
    };
    let mut taken = vec![false; packages.len()];
    let mut todo: Vec<usize> = find(root);
    while let Some(i) = todo.pop() {
        if std::mem::replace(&mut taken[i], true) {
            continue;
        }
        for dep in &packages[i].deps {
            todo.extend(find(dep));
        }
    }
    let mut kept: Vec<&Package> = (0..packages.len()).filter(|&i| taken[i]).map(|i| &packages[i]).collect();
    kept.sort_by(|a, b| (a.name, a.version).cmp(&(b.name, b.version)));
    let mut out = header;
    out.push('\n');
    for p in kept {
        out.push_str("[[package]]\n");
        out.push_str(&p.text);
    }
    out
}

/// The names of the packages in `lock`'s closure of `root`.
pub fn closure_names(lock: &str, root: &str) -> Vec<String> {
    lock_closure(lock, root)
        .lines()
        .filter_map(|l| l.strip_prefix("name = "))
        .map(|n| n.trim_matches('"').to_string())
        .collect()
}

/// Of the workspace's `Cargo.toml` (`manifest`), what can change how this
/// crate and its dependencies are built: the `[workspace]` and
/// `[workspace.package]` sections, every `[profile...]` section, and of
/// `[workspace.dependencies]` the lines of the crates in `names` (this
/// crate's closure). Comments and blank lines are left out; an unrelated
/// crate's line, or the lints, leave it as it is.
pub fn manifest_part(manifest: &str, names: &[String]) -> String {
    let mut out = String::new();
    let mut section = String::new();
    for line in manifest.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t.starts_with('[') {
            section = t.to_string();
            let keep = section == "[workspace]" || section == "[workspace.package]" || section.starts_with("[profile");
            if keep {
                out.push_str(t);
                out.push('\n');
            }
            continue;
        }
        let keep = if section == "[workspace.dependencies]" {
            let key = t.split(['=', ' ']).next().unwrap_or("");
            names.iter().any(|n| n == key)
        } else {
            section == "[workspace]" || section == "[workspace.package]" || section.starts_with("[profile")
        };
        if keep {
            out.push_str(t);
            out.push('\n');
        }
    }
    out
}
