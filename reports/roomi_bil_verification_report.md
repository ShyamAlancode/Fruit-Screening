# Roomi BIL/HDR Verification Report
Generated: 2026-09-10 11:44:47
Source root: `d:\CFRD\DATA\Dataset\roomi_pesticide_hsi`

## GATE 1: Root directory
**PASS** — 79 items

## GATE 2: .bil files
**PASS** — 28 .bil files found

## GATE 3: HDR companions
**PASS** — 0 .bil files missing .hdr

## GATE 4: Header validation
**PASS**
- bands: 300
- wavelength range: 388.9 – 1026.0 nm
- interleave: bil

## GATE 5: ENVI cube loading
**PASS**
- shape: (550, 900, 300)
- dtype: <u2

## GATE 6: Band count
**PASS — 300 bands**

## Final Verdict
**GO**

| Gate | Result |
|---|---|
| gate1 | PASS |
| gate2_bil_count | 28 |
| gate3 | PASS |
| gate4 | PASS |
| gate5 | PASS |
| confirmed_shape | (550, 900, 300) |
| confirmed_bands | 300 |
| gate6 | PASS — 300 bands |