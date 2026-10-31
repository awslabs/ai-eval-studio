# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report,
new feature, correction, or additional documentation, we greatly value feedback and
contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure
we have all the necessary information to effectively respond to your bug report or
contribution.

## Reporting Bugs/Feature Requests

We welcome you to use the issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure
somebody else hasn't already reported the issue. Please try to include as much
information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request,
please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else
   hasn't addressed the problem already.
3. You open an issue to discuss any significant work — we would hate for your time to be
   wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you
   also reformat all the code, it will be hard for us to focus on your change.
3. Ensure the checks below pass locally.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay
   involved in the conversation.

### Checks to run before sending a pull request

```bash
(cd backend  && uv run pytest)      # backend suite
(cd frontend && npm test)           # frontend suite (vitest)
(cd frontend && npm run build)      # eslint + tsc + vite build
./scripts/lint.sh                   # ruff over backend/, infra/, scripts/
./scripts/test-infra.sh             # CDK synth-level tests
```

`npm run build` is the real frontend gate — `tsc` runs there and not under `vitest`, so a
type error passes the test suite and fails the build.

Two conventions to follow when you write code:

* **Comments and docstrings say what the code does and why, never what produced it.** No
  plan, phase, step, milestone or ticket ids, and no references to documents that are not
  in this repository — a reader holding only this repo has to be able to follow every
  comment in it.
* **Nothing may carry an AWS account id, an internal hostname, a private repository URL,
  or real credential material.**

## Finding contributions to work on

Looking at the existing issues is a great way to find something to contribute on. Issues
labelled 'help wanted' are a good place to start.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security issue notifications

See [SECURITY.md](SECURITY.md) for how to report a potential security issue, and for
which known limitations are not vulnerabilities to report. Please do **not** create a
public issue.

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm
the licensing of your contribution.
