(*
Check out https://reference.wolfram.com/language/workflow/SetUpAnEntityStore.html
for more information
*)

store = EntityStore[
  "Pet" -> <|
    "Entities" -> <|
      "Sam" -> <|"BirthDate" -> DateObject[{2016, 7, 23}]|>,
      "Rocky" -> <|"BirthDate" -> DateObject[{2022, 11, 01}]|>,
      "Daisy" -> <|"BirthDate" -> DateObject[{2021, 3, 13}]|>
    |>
  |>
];

Print[store[Entity["Pet", "Sam"], "BirthDate"]];

EntityRegister[store];

Print[Entity["Pet", "Sam"]["BirthDate"]];

EntityUnregister["Pet"];

store = EntityStore[
  "Pet" -> <|
    "Entities" -> <|
      "Sam" -> <|"BirthDate" -> DateObject[{2016, 7, 23}]|>,
      "Rocky" -> <|"BirthDate" -> DateObject[{2022, 11, 01}]|>,
      "Daisy" -> <|"BirthDate" -> DateObject[{2021, 3, 13}]|>
    |>,
    "EntityClasses" -> <|
      "Dogs" -> <|"Entities" -> {"Sam", "Rocky"}|>,
      "Cats" -> <|"Entities" -> {"Daisy"}|>
    |>,
    "Properties" -> <|
      "Age" -> <|"DefaultFunction" -> (DateObject[{2026, 3, 30}] - #["BirthDate"] &)|>
    |>
  |>
];

EntityRegister[store];

Print[EntityClassList["Pet"]];
Print[EntityList[EntityClass["Pet", "Dogs"]]];
Print[Entity["Pet", "Sam"]["Age"]];

Entity["Pet", "Sam"]["BirthDate"] = DateObject[{2017, 2, 9}];
Print[Entity["Pet", "Sam"]["Age"]];

Entity["Pet", "Scout"]["BirthDate"] = DateObject[{2020, 10, 03}];
Print[EntityList[Entity["Pet"]]];

store = EntityStore[
  {
    "Pet" -> <|
      "Entities" -> <|
        "Sam" -> <|"BirthDate" -> DateObject[{2016, 7, 23}]|>,
        "Rocky" -> <|"BirthDate" -> DateObject[{2022, 11, 01}]|>
      |>
    |>,
    "Plant" -> <|
      "Entities" -> <|
        "BigPine" -> <|
          "Location" -> GeoPosition[{40.7128, -74.006}],
          "Species" -> "Pinus strobus"
        |>,
        "PottedFern" -> <|
            "Location" -> GeoPosition[{34.0522, -118.2437}],
            "Species" -> "Nephrolepis exaltata"
        |>
      |>
    |>
  }
];

EntityUnregister["Pet"];
EntityRegister[store];

geoGraphics = GeoGraphics[
  GeoMarker[Entity["Plant", "BigPine"]["Location"]],
  GeoRange -> Quantity[200, "Meter"]
];
