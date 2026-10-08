# v1.0.0 verification

Checked on Windows with Blender 5.2.1 LTS in an isolated factory-startup profile, using the actual extension ZIP and Blender's Install from Disk operator.

Passed: installation and enabling; manifest version 1.0.0; release panel registration; disabling and re-enabling; recovering a known rotated/translated frame from synthetic landmarks; volume estimate from a synthetic 20 mm cube; preservation of that cube; mounting-seat creation; ocular operator registration; loading the bundled FLAME Open reference template.

All Python source files parsed successfully. Blender's extension builder accepted the manifest after shortening its file-permission description. Asset and Python geometry files were compared with the packaged 0.66.81 release; geometry code is unchanged. No case files or patient-specific model files were added.

macOS and Linux have not been runtime-tested. These checks do not cover every workflow, hardware configuration, manufacturing step or upgrade scenario. They do not constitute clinical validation. Airway core removal remains unvalidated. See README.md for limitations.
