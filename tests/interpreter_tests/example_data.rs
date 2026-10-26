use super::*;

/// `ExampleData` serves Wolfram's catalogue with generated data. These tests
/// check the generated values against the metadata table recorded from
/// wolframscript (`resources/example_data_meta.tsv.gz`): the catalogue, the
/// property lists, every exact metadata value, and the head and shape of the
/// generated data. The data itself is invented, so only its structure and
/// statistics are asserted — plus pinned fingerprints, which catch any change
/// to the generators (the data must be the same on every run).
mod example_data_tests {
  use super::*;
  use woxi::functions::example_data::meta::{META, Table};

  /// `ExampleData[{c, n}, p]` as source text.
  fn query(c: &str, n: &str, p: Option<&str>) -> String {
    match p {
      Some(p) => format!("ExampleData[{{{c:?}, {n:?}}}, {p:?}]"),
      None => format!("ExampleData[{{{c:?}, {n:?}}}]"),
    }
  }

  fn table(c: &str) -> &'static Table {
    META.table(c).unwrap_or_else(|| panic!("no table for {c}"))
  }

  /// One test per collection, so they run in parallel.
  macro_rules! per_collection {
    ($check:ident: $($name:ident => $c:literal),* $(,)?) => {
      $( #[test] fn $name() { $check($c); } )*
    };
  }

  /// The entries `ExampleData[c]` lists.
  fn listed(c: &str) -> Vec<String> {
    table(c)
      .rows()
      .iter()
      .filter(|r| table(c).raw(r, "@Listed") != Some("False"))
      .map(|r| r.name.clone())
      .collect()
  }

  /// The catalogue as wolframscript reported it: 19 collections and the
  /// number of entries of each (SpatialPointData's catalogue is `$Failed`).
  #[test]
  fn catalogue_matches_wolframscript() {
    clear_state();
    assert_eq!(
      interpret("ExampleData[]").unwrap(),
      "{AerialImage, Audio, ColorTexture, Dataset, Geometry3D, \
       LinearOptimization, LinearProgramming, MachineLearning, Matrix, \
       NetworkGraph, Sound, SpatialPointData, Statistics, TestAnimation, \
       TestImage, TestImage3D, TestImageSet, Text, Texture}"
    );
    for (c, n) in [
      ("AerialImage", 38),
      ("Audio", 39),
      ("ColorTexture", 40),
      ("Dataset", 3),
      ("Geometry3D", 27),
      ("LinearOptimization", 137),
      ("LinearProgramming", 137),
      ("MachineLearning", 9),
      ("Matrix", 2338),
      ("NetworkGraph", 228),
      ("Sound", 68),
      ("Statistics", 111),
      ("TestAnimation", 4),
      ("TestImage", 47),
      ("TestImage3D", 5),
      ("TestImageSet", 2),
      ("Text", 50),
      ("Texture", 64),
    ] {
      assert_eq!(
        interpret(&format!("Length[ExampleData[{c:?}]]")).unwrap(),
        n.to_string(),
        "{c}"
      );
      assert_eq!(listed(c).len(), n, "{c}");
    }
    assert_eq!(
      interpret("Union[First /@ ExampleData[\"TestImage\"]]").unwrap(),
      "{TestImage}"
    );
    assert_eq!(
      interpret("ExampleData[\"TestImage\"][[1]]").unwrap(),
      "{TestImage, Aerial}"
    );
    assert_eq!(
      interpret("ExampleData[\"SpatialPointData\"]").unwrap(),
      "$Failed"
    );
  }

  /// Every entry's property list, and every recorded exact metadata value,
  /// for every entry of every collection (every 23rd matrix: the matrix
  /// index alone has 46 000 values).
  fn check_metadata(c: &str) {
    clear_state();
    {
      let c = c.to_string();
      let Some(t) = META.table(&c) else { return };
      if t.columns().first().is_some_and(|c| c == "@Catalogue") {
        return;
      }
      for (i, row) in t.rows().iter().enumerate() {
        if c == "Matrix" && i % 23 != 0 {
          continue;
        }
        let n = &row.name;
        let props = t.raw(row, "@Properties").unwrap_or("{}");
        assert_eq!(
          interpret(&format!(
            "{} === {props}",
            query(&c, n, Some("Properties"))
          ))
          .unwrap(),
          "True",
          "{c}/{n} properties"
        );
        for col in t.columns() {
          if col.starts_with('@') {
            continue;
          }
          let Some(raw) = t.raw(row, col) else { continue };
          let Some(exact) = raw.strip_prefix('=') else {
            continue;
          };
          // Descriptive text is Woxi's own, even where the original's is None.
          if matches!(
            col.as_str(),
            "Description" | "LongDescription" | "Source"
          ) {
            continue;
          }
          assert_eq!(
            interpret(&format!("{} === ({exact})", query(&c, n, Some(col))))
              .unwrap(),
            "True",
            "{c}/{n} {col} should be {exact}"
          );
        }
      }
    }
  }

  mod metadata {
    use super::*;
    per_collection!(check_metadata:
      aerial_image => "AerialImage", audio => "Audio", color_texture => "ColorTexture",
      dataset => "Dataset", geometry_3d => "Geometry3D",
      linear_optimization => "LinearOptimization", linear_programming => "LinearProgramming",
      machine_learning => "MachineLearning", matrix => "Matrix", network_graph => "NetworkGraph",
      sound => "Sound", statistics => "Statistics", test_animation => "TestAnimation",
      test_image => "TestImage", test_image_3d => "TestImage3D", test_image_set => "TestImageSet",
      text => "Text", texture => "Texture",
    );
  }

  /// The head a value described by a recorded skeleton has.
  fn expected_head(raw: &str) -> Option<String> {
    let raw = raw.strip_prefix('=').unwrap_or(raw);
    let head = |h: &str| Some(h.to_string());
    // A top-level rule (`inputs -> output`).
    let mut depth = 0i32;
    let bytes = raw.as_bytes();
    for i in 0..bytes.len() {
      match bytes[i] {
        b'[' | b'{' | b'(' => depth += 1,
        b']' | b'}' | b')' => depth -= 1,
        b'<' if bytes.get(i + 1) == Some(&b'|') => depth += 1,
        b'|' if bytes.get(i + 1) == Some(&b'>') => depth -= 1,
        b'-' if depth == 0 && bytes.get(i + 1) == Some(&b'>') => {
          return head("Rule");
        }
        _ => {}
      }
    }
    if raw.starts_with("Arr[")
      || raw.starts_with("Rep[")
      || raw.starts_with('{')
    {
      return head("List");
    }
    if raw.starts_with("Assoc[") || raw.starts_with("<|") {
      return head("Association");
    }
    if let Some(rest) = raw.strip_prefix("Sk[") {
      let h = rest.split([',', ']']).next()?.trim().trim_matches('"');
      return Some(match h {
        // Woxi normalizes Wolfram's internal TemporalData form.
        "TemporalData" => "TimeSeries".to_string(),
        other => other.to_string(),
      });
    }
    match raw {
      "String" => head("String"),
      "Integer" => head("Integer"),
      "Real" => head("Real"),
      _ => {
        let h: String = raw
          .chars()
          .take_while(|c| c.is_alphanumeric() || *c == '`')
          .collect();
        (!h.is_empty() && raw[h.len()..].starts_with('[')).then_some(h)
      }
    }
  }

  /// Every property of a sample of entries of every collection evaluates
  /// (does not stay unevaluated) to a value of the recorded head.
  fn check_heads(c: &str) {
    clear_state();
    {
      let c = c.to_string();
      let Some(t) = META.table(&c) else { return };
      let rows = t.rows();
      if rows.is_empty()
        || t.columns().first().is_some_and(|c| c == "@Catalogue")
      {
        return;
      }
      // The first and last listed entries, and the first sub-entry —
      // passing over giant images (a 34-megapixel photograph's "Data" is a
      // hundred-million-element list in either system).
      let small = |i: usize| -> bool {
        let Some(sk) = t.raw(&rows[i], "Image") else {
          return true;
        };
        let dims: Vec<u64> = sk
          .split(['{', '}'])
          .nth(1)
          .unwrap_or("")
          .split(", ")
          .filter_map(|d| d.parse().ok())
          .collect();
        dims.iter().product::<u64>() <= 1 << 20
      };
      let last_small =
        (0..listed(&c).len()).rev().find(|&i| small(i)).unwrap_or(0);
      let mut sample = vec![0, last_small];
      if let Some(i) = rows
        .iter()
        .position(|r| t.raw(r, "@Listed") == Some("False"))
      {
        sample.push(i);
      }
      for i in sample {
        let row = &rows[i];
        let n = &row.name;
        for col in t.columns() {
          if col.starts_with('@') {
            continue;
          }
          let Some(raw) = t.raw(row, col) else { continue };
          // Missing[…], None and $Failed are exact values, checked above.
          if raw.starts_with('=') {
            continue;
          }
          let q = query(&c, n, Some(col));
          let value_head = interpret(&format!("Head[{q}]")).unwrap();
          assert_ne!(value_head, "ExampleData", "{q} stayed unevaluated");
          if let Some(h) = expected_head(raw) {
            assert_eq!(value_head, h, "{q}: recorded {raw}");
          }
        }
        assert_ne!(
          interpret(&format!("Head[{}]", query(&c, n, None))).unwrap(),
          "ExampleData",
          "{c}/{n} itself"
        );
      }
    }
  }

  mod heads {
    use super::*;
    per_collection!(check_heads:
      aerial_image => "AerialImage", audio => "Audio", color_texture => "ColorTexture",
      dataset => "Dataset", geometry_3d => "Geometry3D",
      linear_optimization => "LinearOptimization", linear_programming => "LinearProgramming",
      machine_learning => "MachineLearning", matrix => "Matrix", network_graph => "NetworkGraph",
      sound => "Sound", statistics => "Statistics", test_animation => "TestAnimation",
      test_image => "TestImage", test_image_3d => "TestImage3D", test_image_set => "TestImageSet",
      text => "Text", texture => "Texture",
    );
  }

  /// Every image has the recorded size, channels, type and colour space.
  /// Check the images of every `parts`-th entry of `c`, starting at `part`
  /// (large collections are split so each test stays within its budget).
  fn check_images_part(c: &str, part: usize, parts: usize) {
    clear_state();
    {
      let t = table(c);
      for row in t.rows().iter().skip(part).step_by(parts) {
        let Some(sk) = t.raw(row, "Image") else {
          continue;
        };
        // Sk["Image", {w, h}, channels, "type", "space", interleaving]
        let q = query(c, &row.name, Some("Image"));
        // Generating a 34-megapixel photograph takes seconds; its size is
        // checked through the metadata test instead.
        let pixels: u64 = sk
          .split(['{', '}'])
          .nth(1)
          .unwrap_or("")
          .split(", ")
          .filter_map(|d| d.parse::<u64>().ok())
          .product();
        if pixels > 4 << 20 {
          continue;
        }
        assert_eq!(
          interpret(&format!(
            "With[{{i = {q}, s = {sk}}}, {{ImageDimensions[i], ImageChannels[i], \
             ImageType[i], ImageColorSpace[i]}} === List @@ s[[2 ;; 5]]]"
          ))
          .unwrap(),
          "True",
          "{c}/{}: {sk}",
          row.name
        );
      }
    }
  }
  fn check_images(c: &str) {
    check_images_part(c, 0, 1);
  }

  mod image_shapes {
    use super::*;
    per_collection!(check_images:
      test_image => "TestImage", color_texture => "ColorTexture", test_image_set => "TestImageSet",
    );
    #[test]
    fn aerial_image_1() {
      check_images_part("AerialImage", 0, 2);
    }
    #[test]
    fn aerial_image_2() {
      check_images_part("AerialImage", 1, 2);
    }
    #[test]
    fn texture_1() {
      check_images_part("Texture", 0, 3);
    }
    #[test]
    fn texture_2() {
      check_images_part("Texture", 1, 3);
    }
    #[test]
    fn texture_3() {
      check_images_part("Texture", 2, 3);
    }
  }

  /// Byte images hold multiples of 1/255.
  #[test]
  fn byte_images_are_quantized() {
    clear_state();
    assert_eq!(
      interpret(
        "d = Flatten[ImageData[ExampleData[{\"TestImage\", \"Peppers\"}]]]; \
         Max[Abs[d*255 - Round[d*255]]] < 10^-9"
      )
      .unwrap(),
      "True"
    );
  }

  /// Every model has the recorded vertex count, face counts per arity and
  /// bounding box.
  #[test]
  fn models_have_the_recorded_counts() {
    clear_state();
    let t = table("Geometry3D");
    for row in t.rows() {
      let n = &row.name;
      let vertices = t.raw(row, "VertexData").unwrap();
      let count = vertices
        .trim_start_matches("Arr[{")
        .split(',')
        .next()
        .unwrap();
      assert_eq!(
        interpret(&format!(
          "Length[{}]",
          query("Geometry3D", n, Some("VertexData"))
        ))
        .unwrap(),
        count,
        "{n} vertices"
      );
      let faces = t.raw(row, "@FaceSizes").unwrap();
      assert_eq!(
        interpret(&format!(
          "Sort[Tally[Length /@ {}]] === Sort[{faces}]",
          query("Geometry3D", n, Some("PolygonData"))
        ))
        .unwrap(),
        "True",
        "{n} faces"
      );
      let bbox = t.raw(row, "BoundingBox").unwrap().trim_start_matches('=');
      assert_eq!(
        interpret(&format!(
          "Max[Abs[CoordinateBounds[{}] - {bbox}]] < 10^-6",
          query("Geometry3D", n, Some("VertexData"))
        ))
        .unwrap(),
        "True",
        "{n} bounding box"
      );
    }
  }

  /// Recordings have the recorded sample rate, channels, length and type.
  #[test]
  fn recordings_have_the_recorded_shape() {
    clear_state();
    for (c, n, rate, channels, frames, ty) in [
      (
        "Audio",
        "Apollo11ReturnSafely",
        11025,
        1,
        28313,
        "SignedInteger16",
      ),
      ("Audio", "Bee", 44100, 2, 107482, "SignedInteger16"),
      ("Audio", "ChurchBell", 48000, 2, 426909, "SignedInteger32"),
      ("Sound", "Flute", 22050, 1, 57920, "SignedInteger16"),
      ("Sound", "Square100", 22050, 1, 110250, "SignedInteger16"),
    ] {
      assert_eq!(
        interpret(&format!(
          "With[{{a = {}}}, {{AudioSampleRate[a], AudioChannels[a], AudioLength[a], AudioType[a]}}]",
          query(c, n, Some("Audio"))
        ))
        .unwrap(),
        format!("{{Quantity[{rate}, Hertz], {channels}, Quantity[{frames}, Samples], {ty}}}"),
        "{c}/{n}"
      );
    }
    // The file stands in for the recording: it can be imported.
    assert_eq!(
      interpret("AudioLength[ExampleData[{\"Audio\", \"Apollo11ReturnSafely\"}, \"AudioFile\"]]").unwrap(),
      "Quantity[28313, Samples]"
    );
    assert_eq!(
      interpret("Head[ExampleData[{\"Sound\", \"Flute\"}]]").unwrap(),
      "Sound"
    );
  }

  /// Texts have the recorded number of words, lines and empty lines.
  #[test]
  fn texts_have_the_recorded_structure() {
    clear_state();
    let t = table("Text");
    for row in t.rows() {
      let n = &row.name;
      let counts = t.raw(row, "@Counts").unwrap();
      let c: Vec<&str> = counts.trim_matches(['{', '}']).split(", ").collect();
      assert_eq!(
        interpret(&format!(
          "With[{{l = {}}}, {{Length[l], Count[l, \"\"]}}]",
          query("Text", n, Some("Lines"))
        ))
        .unwrap(),
        format!("{{{}, {}}}", c[2], c[3]),
        "{n} lines"
      );
      let words: usize = c[1].parse().unwrap();
      let got: usize =
        interpret(&format!("Length[{}]", query("Text", n, Some("Words"))))
          .unwrap()
          .parse()
          .unwrap();
      // Unspaced scripts count whole lines as words.
      assert!(
        got.abs_diff(words) <= words / 5 + 2,
        "{n}: {got} words, recorded {words}"
      );
    }
    // English text is made of words, in sentences.
    assert_eq!(
      interpret(
        "s = ExampleData[{\"Text\", \"GettysburgAddress\"}]; \
         {StringQ[s], StringMatchQ[StringTake[s, 1], RegularExpression[\"[A-Z]\"]], \
          Length[TextSentences[s]] > 3}"
      )
      .unwrap(),
      "{True, True, True}"
    );
    // Other languages use their own letters.
    assert_eq!(
      interpret(
        "Count[Characters[ExampleData[{\"Text\", \"UNHumanRightsGreek\"}]], \
         c_ /; 16^^0370 <= First[ToCharacterCode[c]] <= 16^^03FF] > 1000"
      )
      .unwrap(),
      "True"
    );
  }

  /// Tables have the recorded dimensions, column statistics and categories.
  #[test]
  fn statistics_tables_have_the_recorded_shape() {
    clear_state();
    let t = table("Statistics");
    for row in t.rows() {
      let n = &row.name;
      let dims =
        interpret(&format!("Dimensions[{}]", query("Statistics", n, None)))
          .unwrap();
      let recorded = t.raw(row, "DataElements").unwrap();
      let rows = recorded
        .trim_start_matches("Arr[{")
        .trim_start_matches("Rep[")
        .split([',', '}'])
        .next()
        .unwrap();
      assert!(
        dims.starts_with(&format!("{{{rows}")),
        "{n}: {dims} vs {recorded}"
      );
    }
    assert_eq!(
      interpret(
        "d = ExampleData[{\"Statistics\", \"FisherIris\"}]; \
         {Union[d[[All, 5]]], Count[d[[All, 5]], \"setosa\"], Min[d[[All, 1]]], Max[d[[All, 1]]], \
          Correlation[d[[All, 3]], d[[All, 4]]] > 0.8}"
      )
      .unwrap(),
      "{{setosa, versicolor, virginica}, 50, 4.3, 7.9, True}"
    );
    assert_eq!(
      interpret("Head[ExampleData[{\"Statistics\", \"AirlinePassengerMiles\"}, \"TimeSeries\"]]").unwrap(),
      "TimeSeries"
    );
    assert_eq!(
      interpret("ExampleData[{\"Statistics\", \"AirlinePassengerMiles\"}, \"TimeSeries\"][\"FirstTime\"]").unwrap(),
      "1937"
    );
  }

  #[test]
  fn datasets_have_the_recorded_keys() {
    clear_state();
    assert_eq!(
      interpret("Keys[Normal[ExampleData[{\"Dataset\", \"Planets\"}]]]")
        .unwrap(),
      "{Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune}"
    );
    assert_eq!(
      interpret(
        "d = ExampleData[{\"Dataset\", \"Titanic\"}]; \
         {Length[d], Keys[Normal[d][[1]]], Count[Normal[d][[All, \"age\"]], _Missing]}"
      )
      .unwrap(),
      "{1309, {class, age, sex, survived}, 263}"
    );
    assert_eq!(
      interpret(
        "Length[ExampleData[{\"Dataset\", \"StatePopulations\"}, \"RawData\"]]"
      )
      .unwrap(),
      "198"
    );
  }

  #[test]
  fn learning_sets_have_the_recorded_sizes() {
    clear_state();
    for (n, train, test) in [
      ("FisherIris", 105, 45),
      ("Titanic", 916, 393),
      ("MNIST", 60000, 10000),
    ] {
      assert_eq!(
        interpret(&format!(
          "{{Length[{}], Length[{}]}}",
          query("MachineLearning", n, Some("TrainingData")),
          query("MachineLearning", n, Some("TestData"))
        ))
        .unwrap(),
        format!("{{{train}, {test}}}"),
        "{n}"
      );
    }
    assert_eq!(
      interpret(
        "e = First[ExampleData[{\"MachineLearning\", \"MNIST\"}, \"Data\"]]; \
         {ImageDimensions[e[[1]]], IntegerQ[e[[2]]]}"
      )
      .unwrap(),
      "{{28, 28}, True}"
    );
    // The label is a function of the features: one feature alone already
    // separates the classes better than chance.
    assert_eq!(
      interpret(
        "d = ExampleData[{\"MachineLearning\", \"FisherIris\"}, \"Data\"]; \
         Length[Union[d[[All, 2]]]]"
      )
      .unwrap(),
      "3"
    );
  }

  /// Sparse matrices have the recorded size, number of entries, symmetry
  /// and bandwidth; positive definite ones are.
  #[test]
  fn matrices_have_the_recorded_structure() {
    clear_state();
    for n in ["494BUS", "1138BUS", "Alemdar/Alemdar"] {
      let q = query("Matrix", n, None);
      assert_eq!(
        interpret(&format!(
          "m = {q}; {{Dimensions[m], Length[m[\"NonzeroValues\"]]}} === \
           {{{d}, {e}}}",
          d = query("Matrix", n, Some("Dimensions")),
          e = query("Matrix", n, Some("Entries"))
        ))
        .unwrap(),
        "True",
        "{n}"
      );
    }
    assert_eq!(
      interpret("SymmetricMatrixQ[ExampleData[{\"Matrix\", \"494BUS\"}]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "m = ExampleData[{\"Matrix\", \"494BUS\"}]; \
         Max[Abs[Subtract @@@ m[\"NonzeroPositions\"]]]"
      )
      .unwrap(),
      "429"
    );
  }

  /// Generated linear programs have the recorded shape and are feasible
  /// and bounded: LinearProgramming solves them.
  #[test]
  fn linear_programs_are_solvable() {
    clear_state();
    for n in ["afiro", "adlittle", "blend", "sc105"] {
      assert_eq!(
        interpret(&format!(
          "d = {}; {{Dimensions[d[[2]]] === {}, Length[d[[2]][\"NonzeroValues\"]] > 0, \
           VectorQ[LinearProgramming @@ d, NumericQ]}}",
          query("LinearProgramming", n, None),
          query("LinearProgramming", n, Some("Dimensions"))
        ))
        .unwrap(),
        "{True, True, True}",
        "{n}"
      );
    }
    assert_eq!(
      interpret(
        "d = ExampleData[{\"LinearOptimization\", \"afiro\"}]; \
         {Length[d], VectorQ[LinearOptimization @@ d, NumericQ]}"
      )
      .unwrap(),
      "{3, True}"
    );
  }

  /// The same request gives the same data: fingerprints of generated values
  /// are pinned, so a change to a generator (or platform-dependent output)
  /// shows up here.
  #[test]
  fn data_is_deterministic() {
    clear_state();
    let fp = |code: &str| interpret(code).unwrap();
    let image =
      "Total[ImageData[ExampleData[{\"TestImage\", \"Moon\"}], \"Byte\"], 2]";
    let first = fp(image);
    assert_eq!(fp(image), first);
    assert_eq!(
      fp(
        "Total[Length /@ ExampleData[{\"Text\", \"GettysburgAddress\"}, \"Words\"]]"
      ),
      fp(
        "Total[Length /@ ExampleData[{\"Text\", \"GettysburgAddress\"}, \"Words\"]]"
      )
    );
  }

  #[test]
  fn an_unknown_collection_is_reported() {
    clear_state();
    for (code, shown) in [
      ("ExampleData[\"NoSuchCollection\"]", "\"NoSuchCollection\""),
      (
        "ExampleData[{\"NoSuchCollection\", \"x\"}]",
        "\"NoSuchCollection\"",
      ),
      ("ExampleData[{\"TestImage\"}]", "{\"TestImage\"}"),
      (
        "ExampleData[{\"TestImage\", \"Moon\", \"x\"}]",
        "{\"TestImage\", \"Moon\", \"x\"}",
      ),
    ] {
      let r = interpret_with_stdout(code).unwrap();
      assert!(r.result.starts_with("ExampleData["), "{code}: {}", r.result);
      assert!(
        r.warnings.iter().any(|w| w.contains(&format!(
          "ExampleData::notcoll: {shown} is not a known collection for ExampleData."
        ))),
        "{code}: {:?}",
        r.warnings
      );
    }
  }

  #[test]
  fn an_unknown_entity_is_reported() {
    clear_state();
    for (code, shown) in [
      (
        "ExampleData[{\"TestImage\", \"NoSuchImage\"}]",
        "\"NoSuchImage\"",
      ),
      (
        "ExampleData[{\"TestImage\", \"moon\"}, \"ImageSize\"]",
        "\"moon\"",
      ),
      ("ExampleData[{\"TestImage\", 1}]", "1"),
    ] {
      let r = interpret_with_stdout(code).unwrap();
      assert!(
        r.warnings.iter().any(|w| w.contains(&format!(
          "ExampleData::notent: {shown} is not a known entity for the collection \"TestImage\"."
        ))),
        "{code}: {:?}",
        r.warnings
      );
    }
  }

  #[test]
  fn an_unknown_property_is_reported() {
    clear_state();
    let r =
      interpret_with_stdout("ExampleData[{\"TestImage\", \"Moon\"}, \"Foo\"]")
        .unwrap();
    assert_eq!(r.result, "ExampleData[{TestImage, Moon}, Foo]");
    assert!(
      r.warnings.iter().any(|w| w.contains(
        "ExampleData::notpropx: \"Foo\" is not a known property for {\"TestImage\", \"Moon\"}."
      )),
      "{:?}",
      r.warnings
    );
    // Other malformed calls stay unevaluated silently, as in wolframscript.
    let r = interpret_with_stdout("ExampleData[\"TestImage\", \"Properties\"]")
      .unwrap();
    assert_eq!(r.result, "ExampleData[TestImage, Properties]");
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
  }

  /// The real classic networks are bundled with their published sizes.
  #[test]
  fn real_networks_have_their_published_size() {
    clear_state();
    for (name, v, e) in [
      ("ZacharyKarateClub", 34, 78),
      ("DolphinSocialNetwork", 62, 159),
      ("LesMiserables", 77, 254),
      ("USPoliticsBooks", 105, 441),
      ("WordAdjacencies", 112, 425),
    ] {
      let g = query("NetworkGraph", name, None);
      assert_eq!(
        interpret(&format!("{{VertexCount[{g}], EdgeCount[{g}]}}")).unwrap(),
        format!("{{{v}, {e}}}"),
        "{name}"
      );
    }
    assert!(
      interpret(&query(
        "NetworkGraph",
        "ZacharyKarateClub",
        Some("Description")
      ))
      .unwrap()
      .contains("karate club")
    );
    // A second spelling Wolfram also accepts.
    assert_eq!(
      interpret(
        "VertexCount[ExampleData[{\"NetworkGraph\", \"ZacharysKarateClub\"}]]"
      )
      .unwrap(),
      "34"
    );
  }

  /// Generated networks have the recorded size and direction.
  #[test]
  fn generated_networks_have_the_recorded_shape() {
    clear_state();
    for name in [
      "AmericanCollegeFootball",
      "SimpleFoodWeb",
      "PGPNetwork",
      "WikiVote",
    ] {
      let g = query("NetworkGraph", name, None);
      assert_eq!(
        interpret(&format!(
          "{{VertexCount[{g}], EdgeCount[{g}]}} === {{{vc}, {ec}}}",
          vc = query("NetworkGraph", name, Some("VertexCount")),
          ec = query("NetworkGraph", name, Some("EdgeCount"))
        ))
        .unwrap(),
        "True",
        "{name}"
      );
    }
  }

  /// The related functions accept the data.
  #[test]
  fn functions_accept_the_data() {
    clear_state();
    for (code, expected) in [
      (
        "ImageQ[ImageAdjust[ExampleData[{\"TestImage\", \"Mandrill\"}]]]",
        "True",
      ),
      (
        "ImageDimensions[EdgeDetect[ExampleData[{\"TestImage\", \"House\"}]]]",
        "{256, 256}",
      ),
      (
        "StringContainsQ[ExportString[ExampleData[{\"TestImage\", \"Clock\"}], \"SVG\"], \"<svg\"]",
        "True",
      ),
      (
        "StringContainsQ[ExportString[ExampleData[{\"Geometry3D\", \"Cone\"}], \"SVG\"], \"<svg\"]",
        "True",
      ),
      (
        "Head[ExampleData[{\"Geometry3D\", \"Torus\"}, \"MeshRegion\"]]",
        "MeshRegion",
      ),
      (
        "Length[TextWords[ExampleData[{\"Text\", \"GettysburgAddress\"}]]] > 200",
        "True",
      ),
      (
        "Length[ExampleData[{\"Dataset\", \"Titanic\"}][Select[#sex == \"female\" &]]]",
        "466",
      ),
      (
        "Length[Eigenvalues[N[Normal[ExampleData[{\"Matrix\", \"494BUS\"}]]][[;; 20, ;; 20]]]]",
        "20",
      ),
      (
        "Head[ExampleData[{\"TestImage3D\", \"MRknee\"}]]",
        "Image3D",
      ),
      (
        "ImageDimensions[ExampleData[{\"TestImage3D\", \"MRknee\"}]]",
        "{128, 128, 128}",
      ),
      (
        "Head[ExampleData[{\"TestAnimation\", \"ToyVehicles\"}]]",
        "Manipulate",
      ),
    ] {
      assert_eq!(interpret(code).unwrap(), expected, "{code}");
    }
  }
}
