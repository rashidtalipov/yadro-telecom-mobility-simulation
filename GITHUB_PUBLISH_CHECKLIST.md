# GitHub Publish Checklist

Use this folder as the public repository root:

```text
Yadro/github/oran-lstm-handover
```

Do **not** publish:

```text
Yadro/github/not_for_github
```

## Pre-Publish Checks

Run from `Yadro/github/oran-lstm-handover`:

```bash
rg "/home/|/media/|@|local-user-name|passport|signature|stamp|contract|expert|yandex.ru" .
find . -type f -size +20M -print
find . -type f \( -iname "*.pdf" -o -iname "*.doc" -o -iname "*.docx" -o -iname "*.ppt" -o -iname "*.pptx" \) -print
```

Expected:

- no private local paths;
- no personal documents;
- no signed expert conclusions;
- no raw traces or databases;
- no files larger than 20 MB.

## Repository Initialization

Create a clean repository outside the old `codex-workspace` Git history:

```bash
cd Yadro/github/oran-lstm-handover
git init
git add .
git status --short
git commit -m "Initial public research package"
```

Then create a GitHub repository named:

```text
oran-lstm-handover
```

Add remote and push:

```bash
git remote add origin https://github.com/USERNAME/oran-lstm-handover.git
git branch -M main
git push -u origin main
```

## After Push

Open the repository page and check:

- README renders Mermaid diagrams correctly;
- result tables are readable;
- no private files are visible;
- GitHub link is ready for the YADRO application form.

## License

Choose the final license before making long-term public claims. The C++ scenario files depend on ns-3, so check ns-3 license compatibility.
