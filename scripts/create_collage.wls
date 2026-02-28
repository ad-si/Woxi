headings = {
  Style["CLI", 14, Bold],
  Style["Woxi Studio", 14, Bold],
  Style["JupyterLite", 14, Bold]
};

cli = Import["images/2026-02-28t1010_cli.png"];
woxiStudio = Import["images/2026-02-28t1154_woxi_studio.png"];
jupyterLite = Import["images/2026-02-28t0921_jupyterlite.png"];
imgs = {cli, woxiStudio, jupyterLite};

grid = Grid[
  {headings, imgs},
  Alignment -> Center,
  Spacings -> {0, 1}
];
finalImage = Rasterize[grid, ImageResolution -> 300];
nowStr = DateString[Now, {"Year", "-", "Month", "-", "Day", "T", "Hour", "Minute"}];

Export["images/" <> nowStr <> "_collage.png", finalImage];
