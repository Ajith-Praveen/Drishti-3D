# DRISHTI-3D demo video: voice-over script (about 60 s)

The video (`output/demo/DRISHTI-3D_demo.mp4`, 1080p, 60 s) was rendered from
the real reconstructions and narrated with the macOS "Aman" (English, India)
voice. For a more natural voice, record the lines below over the silent copy
(`output/demo/DRISHTI-3D_demo_silent.mp4`) and swap the audio in.

| Time | On screen | Line |
|---|---|---|
|  0.0– 9.8 s | Title card: logo, DRISHTI-3D, SIH26158 · NTRO | This is DRISHTI-3D, for SIH problem 26158 from NTRO. One drone pass, one video, one measurable 3D model. |
|  9.8–17.5 s | Four keyframes from the DJI video and the flight path | It takes the drone video and its GPS log, picks sharp keyframes, solves every camera, and calibrates the lens by itself. |
| 17.5–28.5 s | Fly-over of the reconstructed town | Depth is measured by stereo in every view and fused in true 3D. Trees come out as trees, houses as houses. An eleven-minute flight, reconstructed in under seven minutes, on a laptop. |
| 28.5–35.4 s | Wipe: 2.5D height map (before) to measured 3D (after) | A simple height map turns everything into prisms. Ours keeps real shapes, and every point carries a confidence. |
| 35.4–43.9 s | Volume and profile measurements on the model; profile chart | Measure right on the model: coordinates, distance, area, stockpile volume and elevation profiles, then export to Google Earth or GIS. |
| 43.9–55.5 s | Results card: 0.29 m, 6.9 min, six formats | On a surveyed flight in Spain, features land within zero point three metres of the national orthophoto. Outputs: OBJ, PLY, LAS, GeoTIFF, glTF and FBX. |
| 55.5–59.7 s | End card: logo, team | DRISHTI-3D. One flight, a 3D model you can trust. |

## Saying it naturally

- Pace: calm and clear, about 150 words a minute. Each line should end a
  little before its scene does (the times above).
- Say "Drishti 3D" as in the Hindi word, "S I H two six one five eight",
  "N T R O", "G P S" and "G I S" letter by letter.
- Formats: "O B J, P L Y, LAS, Geo TIFF, G L T F and F B X".
- "Zero point three metres", not "point three meters"; "eleven-minute flight".
- Smile slightly on the first and last lines; keep the middle factual.

## Recording

1. Play `DRISHTI-3D_demo_silent.mp4` on a laptop and read along on a phone
   (Voice Memos / Recorder app) in a quiet room, phone 20 cm away.
2. Start speaking about half a second after the video starts; leave the
   natural pauses between scenes.
3. Swap the audio (from the project folder):

```bash
ffmpeg -i output/demo/DRISHTI-3D_demo_silent.mp4 -i my_voice.m4a -map 0:v -map 1:a -c:v copy -c:a aac -b:a 160k -shortest output/demo/DRISHTI-3D_demo_voiced.mp4
```

If the voice drifts out of sync, record scene by scene and send the clips;
each scene's start time is in the table.

## Numbers used (all measured, see README and evidence/flight01-benchmark.md)

- 11.4-minute DJI_1001 flight reconstructed in 6.9 min on an M-series laptop;
  8.0 M vertices, 84% directly measured.
- 0.29 m median offset of 13 surveyed features from Spain's national
  orthophoto (IGN PNOA), PinPoint flight01.
- Outputs: OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX (plus textured OBJ/GLB).
