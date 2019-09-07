# ChangeLog

## v0.2.0.1
- Switch to tensorflow 1.14.0.
- Compatibility with stackage LTS-14.4.
- Switch to proto-lens 0.4.

## v0.2.0.0
- Switch to tensorflow 1.9.
- Switch to proto-lens 0.2.2.
- Compatibility with stackage LTS-11.
- Expand the `Rendered` class and add a `ToTensor` class to let more functions
  (gradients, feed, colocateWith) support `ResourceHandle` wrappers like
  `Variable`.
- Add `initializedValue` function for `Variable`.
- Add `TensorFlow.Minimize` module with gradient descent and adam implementations.
- Add more gradient implementations.
- Misc bug fixes.

## v0.1.0.2
- Add extra-lib-dirs for OS X in the Hackage release (#122).

## v0.1.0.1
- Fix the `tensorflow` sdist release by including `c_api.h`.

## v0.1.0.0
- Initial release.
