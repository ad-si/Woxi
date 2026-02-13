# Effect Levels

Each implemented function is classified by its effect level
in [functions.csv](https://github.com/ad-si/Woxi/blob/main/functions.csv).

| Name          | Deterministic | Side effects        | Example        |
| ------------- | ------------- | ------------------- | -------------- |
| `pure`        | Yes           | No                  | `Plus[]`       |
| `contextual`  | No            | No                  | `DateString[]` |
| `effectful`   | No            | Yes                 | `URLRead[]`    |
| `stateful`    | No            | Yes (mutates state) | `count++`      |
