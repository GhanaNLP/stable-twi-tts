# Publishing to PyPI

`.github/workflows/release.yml` builds, checks and uploads the package when a `v*` tag is pushed.
It authenticates with **Trusted Publishing** (OIDC), so no API token is ever stored in this
repository — PyPI verifies that the upload came from this repo and this workflow, and nothing
else can impersonate it. That is why the steps below are done on PyPI's website rather than by
pasting a token into GitHub secrets.

Everything in the repository is ready. What remains needs an account and cannot be automated.

## What you need to do

### 1. Accounts, with 2FA

Create accounts on **[pypi.org](https://pypi.org/account/register/)** and on
**[test.pypi.org](https://test.pypi.org/account/register/)** (a separate registration — the two
share no data). PyPI has required two-factor authentication for uploads since 2024, so enable it
on both. Use an address the organisation keeps, not a personal one; the account owns the name.

### 2. Claim the name with a first upload

`stable-twi-tts` is unclaimed as of writing, but a name belongs to whoever uploads first, so this
is the step worth not delaying. There is no way to reserve a name without uploading.

You cannot configure a trusted publisher for a project that does not exist yet, so the *first*
release has to be a manual upload. Once it exists, every later release goes through the workflow.

```bash
git clone https://github.com/GhanaNLP/stable-twi-tts && cd stable-twi-tts
pip install build twine
python -m build                       # writes dist/*.whl and dist/*.tar.gz
twine check dist/*                    # both should say PASSED

twine upload --repository testpypi dist/*      # rehearse on TestPyPI first
twine upload dist/*                            # then the real one
```

`twine` will ask for a username and password: enter `__token__` as the username and an API token
as the password, generated at [pypi.org/manage/account/token](https://pypi.org/manage/account/token/).
Scope that first token to "entire account" — a project-scoped token cannot exist before the
project does. **Delete it after step 3**, since the workflow will not need it.

Verify the upload from a clean environment before announcing anything:

```bash
python -m venv /tmp/v && /tmp/v/bin/pip install stable-twi-tts
/tmp/v/bin/stable-twi-tts --help
```

### 3. Configure the trusted publisher

At **https://pypi.org/manage/project/stable-twi-tts/settings/publishing/**, add a GitHub publisher:

| field | value |
|---|---|
| Owner | `GhanaNLP` |
| Repository name | `stable-twi-tts` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name must match `environment: pypi` in the workflow — PyPI checks it, and a
mismatch fails the upload with an unhelpful error. Optionally create that environment under the
repository's *Settings → Environments* and add yourself as a required reviewer, which makes every
release pause for a human click.

Then revoke the account-scoped token from step 2.

### 4. Release, from then on

```bash
# bump `version` in pyproject.toml, commit
git tag v0.2.0 && git push origin v0.2.0
```

The workflow refuses to run if the tag and `pyproject.toml` disagree. That check exists because
**PyPI does not allow re-uploading a version** — not even after deleting it — so an accidental
`v0.2.0` carrying `0.1.0` would burn the number permanently.

## What I already changed to make this publishable

- **Removed the `twi` extra.** It pointed at `git+https://…` for ghana-g2p, and PyPI rejects any
  distribution whose metadata contains a direct URL: *"Can't have direct dependency"*. The upload
  would have failed. ghana-g2p is now a documented separate install, in the README and in the
  error message `g2p.py` raises when it is missing.
- **Added `[project.urls]`, `keywords`, `classifiers` and `authors`**, so the project page shows
  something other than a bare description.
- **Added the release workflow**, with a tag/version consistency check, `twine check`, and an
  install-and-import smoke test of the built wheel before it uploads anything.

Verified locally: `python -m build` succeeds, `twine check` passes on both artifacts, and the
wheel's metadata carries no direct-URL dependencies.

## Two things worth deciding before you upload

**The Twi front-end will not work out of a plain `pip install`.** A user who runs
`pip install stable-twi-tts` gets ONNX inference, the CLI and the web UI, but Twi text raises a
`PhonemeError` telling them to install ghana-g2p. That is honest and the message is clear, but it
is a rough first impression. The clean fix is to publish `ghana-g2p` (and `africa-g2p`, whose
duplicate `data/` directory breaks its own wheel build) to PyPI as well, after which `twi` can go
back to being a normal extra. Worth doing, but it is a separate piece of work on another repo.

**The version is `0.1.0`.** Under the usual reading that signals "expect breaking changes", which
is accurate right now — the front-end conventions and `voices.json` schema are both young. Keep
it unless you intend the API to be stable.
