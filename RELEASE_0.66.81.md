# Anaplast Studio 0.66.81 — wax sizing from the Sculpt

Save your case and close Blender before running Anaplast_Studio_Setup_0.66.81.exe. The ZIP is available for manual installation.

Wax reservoirs now use the Sculpt for automatic sizing. Leave Fill estimate at 0: the builder estimates the Sculpt volume and adds 20%, then sizes the wax reserve using the existing Extra wax (%) control. The default reserve remains 10% of that generous fill estimate. No recorded mold volume is required, and missing volume records no longer cancel the airway reservoir build. A positive Fill estimate overrides the automatic estimate.

For a closed Sculpt the estimate uses its enclosed volume. An open Sculpt uses a deliberately generous bounding-box estimate. If no usable Sculpt is selected, a clearly labeled nominal 12 mL estimate is used instead of stopping the build for a volume check. These are reservoir-sizing estimates, not exact silicone consumption measurements. The separate Export & Save volume report still uses mold measurements.

Applies to airway + wax/air-return, wax reservoirs, and combined wax + ocular pedestal inserts. Source Sculpt and fitting scan geometry are unchanged. The built insert report records the estimate and its source. Seat orientation and one/two-seat controls from 0.66.80, plus the corneal correction, are retained.

In Mold > Interchangeable inserts, choose your airway, keep Include wax reservoir + air return enabled, leave Fill estimate at 0, and use Build / update all inserts.

The nasal airway support now anchors into the actual mounting plug instead of assuming solid material 2 mm below the mark. This fixes a disconnected support on the saved recessed nasal base while retaining the airway core shape.

For the nasal airway with two seats, choose Reservoir seat and Vent seat. One passage goes through each seat: wax enters through the first cup and air/wax overflow returns through the second. The ocular pedestal and ordinary wax reservoir arrangements are unchanged. Both connected and separate pieces are supported. A one-seat setup keeps its combined feed and overflow cups. Restoring a mold retains the previous mounting placement for the next build, keeping airway-opening marks separate from seat placement. Existing seats still require confirmation if enlargement is needed.
