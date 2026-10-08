# Anaplast Studio v1.0.0

Open-source Blender tools for digital workflows in anaplastology and maxillofacial prosthetics.

## Release status
Experimental preview — not clinically validated

Anaplast Studio is an open-source Blender add-on under development for digital workflows in anaplastology and maxillofacial prosthetics.

This release is provided for education and technical evaluation using synthetic demonstration models. Its safety, accuracy and suitability for patient treatment have not been clinically validated.

Qualified clinicians must exercise independent clinical judgment and critically assess outputs. Clinical judgment does not replace the validation or regulatory requirements applicable to patient treatment.

## Download v1.0.0

Use the **[v1.0.0 release downloads](https://github.com/Anaplaststudio/anaplast-studio/releases/tag/v1.0.0)**.

- **Windows:** double-click `Anaplast_Studio_Setup_1.0.0_Windows.exe`. Read the release notice, check the acknowledgment box, then select **Back up and install**.
- **macOS:** unzip the installer and open the included app. This is an unsigned, untested preview.
- **Linux:** run `sh Anaplast_Studio_Setup_1.0.0_Linux.run`. This is an untested preview; it uses a graphical acknowledgment where Zenity is available, otherwise terminal prompts.

Blender 5.2+ must already be installed. Close Blender before running an installer. Installers show the full experimental-release notice and require acknowledgment. Windows checks passed in an isolated profile, including backup/reinstall and enablement after restart. Installers are unsigned; the Mac app is not notarized. OS security policies may prevent launch.

## Manual installation — Windows, macOS and Linux

The same `Anaplast_Studio-1.0.0.zip` installs on all three operating systems. It is a Blender extension, not a standalone application. No separate EXE, DMG or Linux installer is required.

1. Install Blender 5.2 or newer. This minimum is required by the distance-grid tools; older Blender versions are not supported.
2. Back up your demonstration project. Leave the downloaded ZIP compressed (on macOS, disable automatic ZIP extraction in your browser if needed).
3. In Blender, open Edit → Preferences → Add-ons. Open the menu at the top right and choose Install from Disk. Select the ZIP and enable Anaplast Studio.
4. Open the 3D Viewport sidebar with N and select the Anaplast tab. The release information panel displays v1.0.0.
5. Begin with a synthetic demonstration model. Restart Blender after updating an existing installation. If an older copy was installed through a different repository, disable that copy before enabling v1.0.0; do not run two copies together.

On MacBooks, use the Blender build matching your processor (Apple Silicon or Intel). The add-on itself uses the same ZIP. Blender's own hardware requirements still apply.

Platform verification: Windows installation and representative synthetic checks are recorded in VALIDATION.md. macOS and Linux have not been runtime-tested for this release. The source uses Blender's Python API, NumPy included with Blender, and Python's standard library; there are no bundled platform-specific binaries or external Python packages.

## Included workflows

- Landmark-based orientation, mirroring and alignment.
- Sculpt preparation, surface detail and prototype tools.
- Mold construction, registration features and interchangeable inserts.
- Nasal template tools and experimental airway/insert geometry.
- Ocular design and reference-photograph workflows.

The v1 release packages development build 0.66.81. Geometry algorithms have not been changed for this release. See RELEASE_1.0.0.md and RELEASE_0.66.81.md.

## Known limitations

Geometric closure and synthetic checks do not establish clinical fit, biocompatibility, safe airway dimensions, effective ventilation, surgical-guide accuracy or clinical performance. Airway core removal has not been validated. Manufacturing parameters and material estimates require independent verification. Nasal template geometry does not reconstruct a missing patient-specific airway. Ocular fitting surfaces are not impression-derived.

## Source and licensing

The installation ZIP contains the complete Python source. Code is licensed under GPL-3.0-or-later; see LICENSE.txt. Bundled model data and reference photographs have separate terms listed in THIRD_PARTY.md and their accompanying attribution files. GPL licensing does not establish clinical suitability or regulatory approval.

## Privacy and contributions

Use synthetic examples for public demonstrations, bug reports and forum discussions. Do not share patient scans, identifiable photographs, names or medical record numbers. Inspect exported findings reports before sharing: they can include case settings, object names and local paths. Only load model files from trusted sources; legacy pickle-based models can execute code when opened.

Report reproducible issues with your operating system, Blender version, add-on version, steps and a synthetic example at support@anaplaststudio.com. Code patches, documentation and macOS/Linux testing are welcome. Open an issue in this repository using a synthetic example, or submit a pull request. The release ZIP also includes the source.

## Working from the source repository

The Python source and smaller reference assets are tracked here. The prepared FLAME model (about 35 MB) is included in the complete release ZIP rather than this source tree. To reproduce the complete package, run `python build_release.py --release-zip /path/to/Anaplast_Studio-1.0.0.zip`. The script verifies the release ZIP checksum, copies only that reference model into the package, and builds the current source tree. Use the complete release ZIP or an installer for normal installation; GitHub's automatic source archive does not contain the large model.

Installer source is supplied separately as `Anaplast_Studio_Installer_Source-1.0.0.zip` on the same release. All reference-asset attribution and license terms must remain with redistributed packages.
