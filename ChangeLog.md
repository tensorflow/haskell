# ChangeLog

## Upcoming (v0.2.0.0)
- Expand the `Rendered` class and add a `ToTensor` class to let more functions
  (gradients, feed, colocateWith) support `ResourceHandle` wrappers like
  `Variables`.
- Add `initializedValue` function for `Variable`.

## v0.1.0.2
- Add extra-lib-dirs for OS X in the Hackage release (#122).

## v0.1.0.1
- Fix the `tensorflow` sdist release by including `c_api.h`.

## v0.1.0.0
- Initial release.
