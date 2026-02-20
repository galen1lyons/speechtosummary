"""
sync_project.py — Audit and fix import/file mismatches in src/.

Run this AFTER check_project.py passes.
It reads every .py file in src/, traces every "from .xxx import yyy" line,
checks whether the target file and name actually exist, and reports
exactly what is broken and what is in sync.

It does NOT modify any file automatically.  It prints the exact commands
or edits you need.

Usage:
    python sync_project.py
"""

import ast
import sys
from pathlib import Path

# ── colour helpers ──
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"

ROOT = Path.cwd()
SRC  = ROOT / "src"

if not SRC.is_dir():
    print(red("❌  src/ not found.  Run this from inside speechtosummary/"))
    sys.exit(1)

# ─────────────────────────────────────────────
# STEP 1 — enumerate every .py file in src/
# ─────────────────────────────────────────────
src_files = sorted(SRC.glob("*.py"))
src_names = {f.stem for f in src_files}          # e.g. {"__init__", "config", ...}

print(bold("\n📂  Files found in src/"))
for f in src_files:
    print(f"      {f.name}")

# ─────────────────────────────────────────────
# STEP 2 — parse every file: extract all
#           "from .module import name" statements
# ─────────────────────────────────────────────
# Map:  filename  →  list of (module_stem, [imported_names])
file_imports = {}       # what each file asks for

for pyfile in src_files:
    source = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(red(f"  ❌  Syntax error in {pyfile.name}: {e}"))
        continue

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # only relative imports (level >= 1) matter here
            if node.level and node.level >= 1 and node.module:
                names = [alias.name for alias in node.names]
                imports.append((node.module, names))
    file_imports[pyfile.name] = imports

# ─────────────────────────────────────────────
# STEP 3 — parse every file: collect what each
#           module actually DEFINES (top-level)
# ─────────────────────────────────────────────
# Map:  module_stem  →  set of defined names
module_exports = {}

for pyfile in src_files:
    source = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    defined = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)

    module_exports[pyfile.stem] = defined

# ─────────────────────────────────────────────
# STEP 4 — cross-reference: does every import
#           resolve to a real file + real name?
# ─────────────────────────────────────────────
print(bold("\n🔍  Cross-referencing imports …\n"))

all_ok      = True
missing_files = set()    # modules that are imported but have no .py
missing_names = []       # (importer, module, name)  — file exists but name missing

for filename, imports in sorted(file_imports.items()):
    if not imports:
        continue

    print(bold(f"  📄 {filename}"))

    for mod_stem, names in imports:
        file_exists = mod_stem in src_names

        if not file_exists:
            # the target module .py doesn't exist at all
            print(f"      {red('❌')}  from .{mod_stem} import …")
            print(f"            {red(f'src/{mod_stem}.py does NOT exist')}")
            for n in names:
                print(f"              missing: {n}")
            missing_files.add(mod_stem)
            all_ok = False
            continue

        # file exists — check each imported name
        exported = module_exports.get(mod_stem, set())
        all_names_ok = True

        for n in names:
            if n in exported:
                print(f"      {green('✅')}  from .{mod_stem} import {n}")
            else:
                print(f"      {red('❌')}  from .{mod_stem} import {n}")
                print(f"            {red(f'{n} is not defined in src/{mod_stem}.py')}")
                missing_names.append((filename, mod_stem, n))
                all_ok = False
                all_names_ok = False

    print()

# ─────────────────────────────────────────────
# STEP 5 — check for DEAD files (exist but
#           nobody imports them)
# ─────────────────────────────────────────────
print(bold("🗑️   Dead files (not imported by anything)"))

all_imported_modules = set()
for imports in file_imports.values():
    for mod_stem, _ in imports:
        all_imported_modules.add(mod_stem)

# __init__ and pipeline are entry-points, not "imported" by siblings
entry_points = {"__init__", "pipeline"}

for stem in src_names - all_imported_modules - entry_points:
    print(f"      {yellow('⚠️ ')}  src/{stem}.py  — nothing imports it")

print()

# ─────────────────────────────────────────────
# STEP 6 — check __all__ in __init__.py lines up
# ─────────────────────────────────────────────
print(bold("📋  __init__.py  __all__ vs actual imports"))
init_path = SRC / "__init__.py"
if init_path.is_file():
    source = init_path.read_text(encoding="utf-8")
    tree   = ast.parse(source)

    # grab __all__ list if present
    all_names = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    # evaluate the literal list
                    try:
                        all_names = ast.literal_eval(node.value)
                    except Exception:
                        pass

    # grab every name that is actually imported at the top
    actually_imported = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                actually_imported.add(alias.asname or alias.name)

    if all_names is not None:
        all_set = set(all_names)
        in_all_not_imported = all_set - actually_imported - {"__version__"}
        imported_not_in_all = actually_imported - all_set

        if in_all_not_imported:
            print(f"      {red('❌')}  In __all__ but NOT imported: {in_all_not_imported}")
            all_ok = False
        if imported_not_in_all:
            print(f"      {yellow('⚠️ ')}  Imported but NOT in __all__: {imported_not_in_all}")
        if not in_all_not_imported and not imported_not_in_all:
            print(f"      {green('✅')}  __all__ matches imports perfectly")
    else:
        print(f"      {yellow('⚠️ ')}  No __all__ defined")
else:
    print(f"      {red('❌')}  __init__.py missing")

print()

# ─────────────────────────────────────────────
# STEP 7 — SUMMARY + ACTION ITEMS
# ─────────────────────────────────────────────
print("─" * 60)

if all_ok:
    print(bold(green("\n🎉  All imports are in sync.  Nothing to fix.\n")))
    print("    Your next step: explore ILMU / MaLLaM on HuggingFace.")
    print("    Run:  python research_my_models.py")
else:
    print(bold(red(f"\n🔧  Issues found — exact fixes below:\n")))

    if missing_files:
        print(bold("  A) Missing source files"))
        print("     These modules are imported but the .py file does not exist.\n")
        for mod in sorted(missing_files):
            print(f"     src/{mod}.py  — does not exist")
            # figure out who wants it
            for fname, imports in file_imports.items():
                for m, names in imports:
                    if m == mod:
                        print(f"       imported by {fname}: {names}")
        print()
        print("     Options:")
        print("       • If you have the file somewhere (e.g. src_broken_backup/),")
        print("         copy it in:   cp src_broken_backup/{mod}.py src/")
        print("       • If you don't need it, remove the import line from the")
        print("         file that references it.")
        print()

    if missing_names:
        print(bold("  B) Names imported but not defined in the target module"))
        for importer, mod, name in missing_names:
            print(f"     {importer}  does  'from .{mod} import {name}'")
            print(f"       but {name} is not defined in src/{mod}.py")
            print(f"       → either add {name} to src/{mod}.py, or remove the import")
        print()

print("─" * 60)
