// Tests for host-derived symbols (`$UserName`, `$MachineName`,
// `$HomeDirectory`, …). Expected values are computed dynamically
// from the same OS facilities the implementation in
// `src/evaluator/listable.rs` consults (env vars, `gethostname`,
// `current_dir`, `cfg!(target_os = …)`), so the assertions hold on
// any host without a controlled capture environment.
//
// What the tests catch even though the expected value is host-derived:
//   * the symbol resolves at all (no Identifier passthrough),
//   * the result is a String of the expected shape (quoting, trailing
//     slash policy, short-hostname stripping, …),
//   * the implementation reads the right source (e.g. `$UserName`
//     from `$USER`, not `$LOGNAME`),
//   * platform branches stay in sync (e.g. `$OperatingSystem` returns
//     "MacOSX" on macOS and "Unix" on Linux).

use super::*;

mod machine_specific {
  use super::*;

  fn normalise(s: &str) -> String {
    s.chars()
      .filter(|c| !c.is_whitespace() && *c != '"')
      .collect()
  }

  fn assert_eval(input: &str, expected: &str) {
    clear_state();
    let actual = match interpret(input) {
      Ok(s) => s,
      Err(e) => panic!(
        "Woxi returned error: {:?}\n  input:    {}\n  expected: {}",
        e, input, expected
      ),
    };
    if normalise(&actual) != normalise(expected) {
      panic!(
        "output mismatch\n  input:    {}\n  expected: {}\n  actual:   {}",
        input, expected, actual
      );
    }
  }

  fn host_home() -> String {
    std::env::var("HOME")
      .or_else(|_| std::env::var("USERPROFILE"))
      .expect("HOME (or USERPROFILE) must be set in the test environment")
  }

  fn host_user() -> String {
    std::env::var("USER")
      .or_else(|_| std::env::var("USERNAME"))
      .expect("USER (or USERNAME) must be set in the test environment")
  }

  /// Short hostname, stripped of trailing `.local` / `.lan` etc, to
  /// mirror `$MachineName` in `src/evaluator/listable.rs`.
  #[cfg(unix)]
  fn host_machine_name() -> String {
    let mut buf = [0u8; 256];
    let ret = unsafe {
      libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len())
    };
    assert_eq!(ret, 0, "gethostname() failed");
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let host =
      std::str::from_utf8(&buf[..len]).expect("hostname is not valid UTF-8");
    host.split('.').next().unwrap_or(host).to_string()
  }

  fn host_initial_dir() -> String {
    std::env::current_dir()
      .expect("current_dir() failed")
      .to_string_lossy()
      .into_owned()
  }

  fn host_temp_dir() -> String {
    let tmp = std::env::temp_dir();
    let canon = std::fs::canonicalize(&tmp).unwrap_or(tmp);
    let mut s = canon.to_string_lossy().into_owned();
    while s.len() > 1 && s.ends_with('/') {
      s.pop();
    }
    s
  }

  #[test]
  fn environment_home() {
    assert_eval(r#"Environment["HOME"]"#, &format!(r#""{}""#, host_home()));
  }

  #[cfg(unix)]
  #[test]
  fn machine_name() {
    assert_eval(r#"$MachineName"#, &format!(r#""{}""#, host_machine_name()));
  }

  #[test]
  fn user_name() {
    assert_eval(r#"$UserName"#, &format!(r#""{}""#, host_user()));
  }

  #[test]
  fn home_directory() {
    assert_eval(r#"$HomeDirectory"#, &format!(r#""{}""#, host_home()));
  }

  #[test]
  fn user_base_directory() {
    let sub = if cfg!(target_os = "macos") {
      "Library/Wolfram"
    } else if cfg!(target_os = "windows") {
      "AppData\\Roaming\\Wolfram"
    } else {
      ".Wolfram"
    };
    let expected =
      format!(r#""{}/{}""#, host_home().trim_end_matches('/'), sub);
    assert_eval(r#"$UserBaseDirectory"#, &expected);
  }

  #[test]
  fn base_directory() {
    let root = if cfg!(target_os = "macos") {
      "/Library/Wolfram"
    } else if cfg!(target_os = "windows") {
      "C:\\ProgramData\\Wolfram"
    } else {
      "/usr/share/Wolfram"
    };
    assert_eval(r#"$BaseDirectory"#, &format!(r#""{}""#, root));
  }

  #[test]
  fn initial_directory() {
    assert_eval(
      r#"$InitialDirectory"#,
      &format!(r#""{}""#, host_initial_dir()),
    );
  }

  #[test]
  fn temporary_directory() {
    assert_eval(
      r#"$TemporaryDirectory"#,
      &format!(r#""{}""#, host_temp_dir()),
    );
  }

  #[test]
  fn parent_directory() {
    let parent = std::path::PathBuf::from(host_initial_dir())
      .parent()
      .expect("current_dir has no parent")
      .to_string_lossy()
      .into_owned();
    assert_eval(r#"ParentDirectory[]"#, &format!(r#""{}""#, parent));
  }

  #[test]
  fn expand_file_name() {
    let expected = format!(
      r#""{}/ExampleData/sunflowers.jpg""#,
      host_initial_dir().trim_end_matches('/'),
    );
    assert_eval(r#"ExpandFileName["ExampleData/sunflowers.jpg"]"#, &expected);
  }

  #[test]
  fn operating_system() {
    let os = if cfg!(target_os = "macos") {
      "MacOSX"
    } else if cfg!(target_os = "linux") {
      "Unix"
    } else if cfg!(target_os = "windows") {
      "Windows"
    } else {
      "Unknown"
    };
    assert_eval(r#"$OperatingSystem"#, &format!(r#""{}""#, os));
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn greater_1() {
    assert_case(r#"$Assumptions = { x > 0 }"#, r#"{x > 0}"#);
  }
  #[test]
  fn assuming() {
    assert_case(
      r#"$Assumptions = { x > 0 }; Assuming[y>0, ConditionalExpression[y x^2, y>0]//Simplify]"#,
      r#"x^2*y"#,
    );
  }
  #[test]
  fn conditional_expression() {
    assert_case(
      r#"ConditionalExpression[x^2, True]; ConditionalExpression[x^2, False]; f = ConditionalExpression[x^2, x>0]; f /. x -> 2; f /. x -> -2; $Assumptions = x > 0; ConditionalExpression[x ^ 2, x>0]//Simplify"#,
      r#"x ^ 2"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(r#"$TraceBuiltins = True"#, r#"True"#);
  }
  #[test]
  fn set_2() {
    assert_case(r#"$TraceBuiltins = True"#, r#"True"#);
  }
  #[test]
  fn set_3() {
    assert_case(
      r#"$TraceBuiltins = True; $TraceBuiltins = False"#,
      r#"False"#,
    );
  }
  #[test]
  fn symbol_literal() {
    assert_case(
      r#"$TraceBuiltins = True; $TraceBuiltins = False; x"#,
      r#"x"#,
    );
  }
  #[test]
  fn set_4() {
    assert_case(r#"$TraceEvaluation = True"#, r#"True"#);
  }
  #[test]
  fn plus_1() {
    assert_case(r#"$TraceEvaluation = True; a + a"#, r#"2*a"#);
  }
  #[test]
  fn set_5() {
    assert_case(
      r#"$TraceEvaluation = True; a + a; $TraceEvaluation = False"#,
      r#"False"#,
    );
  }
  #[test]
  fn trace_evaluation() {
    assert_case(
      r#"$TraceEvaluation = True; a + a; $TraceEvaluation = False; $TraceEvaluation"#,
      r#"False"#,
    );
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"$TraceEvaluation = True; a + a; $TraceEvaluation = False; $TraceEvaluation; a + a"#,
      r#"2*a"#,
    );
  }
  #[test]
  fn set_6() {
    assert_case(
      r#"$TraceEvaluation = True; a + a; $TraceEvaluation = False; $TraceEvaluation; a + a; $TraceEvaluation = x"#,
      r#"x"#,
    );
  }
  #[test]
  fn machine() {
    assert_case(r#"$Machine"#, r#"$Machine"#);
  }
  #[test]
  fn max_length_int_string_conversion() {
    assert_case(
      r#"$MaxLengthIntStringConversion"#,
      r#"$MaxLengthIntStringConversion"#,
    );
  }
  #[test]
  fn divide() {
    assert_case(
      r#"$MaxLengthIntStringConversion; 500! //ToString//StringLength"#,
      r#"1135"#,
    );
  }
  #[test]
  fn factorial() {
    assert_case(
      r#"$MaxLengthIntStringConversion; 500! //ToString//StringLength; $MaxLengthIntStringConversion = 640; 500!"#,
      r#"1220136825991110068701238785423046926253574342803192842192413588385845373153881997605496447502203281863013616477148203584163378722078177200480785205159329285477907571939330603772960859086270429174547882424912726344305670173270769461062802310452644218878789465754777149863494367781037644274033827365397471386477878495438489595537537990423241061271326984327745715546309977202781014561081188373709531016356324432987029563896628911658974769572087926928871281780070265174507768410719624390394322536422605234945850129918571501248706961568141625359056693423813008856249246891564126775654481886506593847951775360894005745238940335798476363944905313062323749066445048824665075946735862074637925184200459369692981022263971952597190945217823331756934581508552332820762820023402626907898342451712006207714640979456116127629145951237229913340169552363850942885592018727433795173014586357570828355780158735432768888680120399882384702151467605445407663535984174430480128938313896881639487469658817504506926365338175055478128640000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"#,
    );
  }
  #[test]
  fn history_length() {
    assert_case(r#"$HistoryLength"#, r#"Infinity"#);
  }
  #[test]
  fn integer_literal_1() {
    assert_case(r#"$HistoryLength; $HistoryLength = 1; 42"#, r#"42"#);
  }
  #[test]
  fn expression_1() {
    assert_case(r#"$HistoryLength; $HistoryLength = 1; 42; %"#, r#"Out[0]"#);
  }
  #[test]
  fn expression_2() {
    assert_case(
      r#"$HistoryLength; $HistoryLength = 1; 42; %; %%"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn integer_literal_2() {
    assert_case(
      r#"$HistoryLength; $HistoryLength = 1; 42; %; %%; $HistoryLength = 0; 42"#,
      r#"42"#,
    );
  }
  #[test]
  fn expression_3() {
    assert_case(
      r#"$HistoryLength; $HistoryLength = 1; 42; %; %%; $HistoryLength = 0; 42; %"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn line_1() {
    assert_case(r#"$Line"#, r#"1"#);
  }
  #[test]
  fn line_2() {
    assert_case(r#"$Line; $Line"#, r#"1"#);
  }
  #[test]
  fn times_1() {
    assert_case(r#"$Line; $Line; $Line = 12; 2 * 5"#, r#"10"#);
  }
  #[test]
  fn out() {
    assert_case(r#"$Line; $Line; $Line = 12; 2 * 5; Out[13]"#, r#"Out[13]"#);
  }
  #[test]
  fn set_7() {
    assert_case(
      r#"$Line; $Line; $Line = 12; 2 * 5; Out[13]; $Line = -1"#,
      r#"-1"#,
    );
  }
  #[test]
  fn iteration_limit() {
    assert_case(r#"$IterationLimit"#, r#"4096"#);
  }
  #[test]
  fn times_2() {
    assert_case(
      r#"$MaxMachineNumber * $MinMachineNumber"#,
      r#"3.9999999999999996"#,
    );
  }
  #[test]
  fn box_forms() {
    assert_case(r#"$BoxForms"#, r#"{StandardForm, TraditionalForm}"#);
  }
  #[test]
  fn output_forms() {
    // Wolframscript-matched expectation. The mathics original tacks on ten
    // `Removed["MyBoxForm"]` entries that are leftover state from earlier
    // mathics tests — wolframscript with a fresh kernel returns just the
    // standard list, which is what Woxi also produces.
    assert_case(
      r#"$OutputForms"#,
      r#"{InputForm, OutputForm, TextForm, CForm, Short, Shallow, MatrixForm, TableForm, TreeForm, FullForm, NumberForm, EngineeringForm, ScientificForm, QuantityForm, DecimalForm, PercentForm, PaddedForm, AccountingForm, BaseForm, DisplayForm, StyleForm, FortranForm, ScriptForm, MathMLForm, TeXForm, StandardForm, TraditionalForm}"#,
    );
  }
  #[test]
  fn print_forms() {
    assert_case(
      r#"$PrintForms"#,
      r#"{InputForm, OutputForm, TextForm, CForm, FortranForm, ScriptForm, MathMLForm, TeXForm, StandardForm, TraditionalForm}"#,
    );
  }
  #[test]
  fn max_precision() {
    assert_case(r#"$MaxPrecision"#, r#"Infinity"#);
  }
  #[test]
  fn machine_epsilon() {
    assert_case(r#"$MachineEpsilon"#, r#"2.220446049250313*^-16"#);
  }
  #[test]
  fn minus() {
    assert_case(
      r#"$MachineEpsilon; x = 1.0 + {0.4, 0.5, 0.6} $MachineEpsilon; x - 1"#,
      r#"{0., 0., 2.220446049250313*^-16}"#,
    );
  }
  #[test]
  fn machine_precision_1() {
    assert_case(r#"$MachinePrecision"#, r#"15.954589770191003"#);
  }
  #[test]
  fn min_precision() {
    assert_case(r#"$MinPrecision"#, r#"0"#);
  }
  #[test]
  fn character_encoding() {
    assert_case(r#"$CharacterEncoding"#, r#""UTF8""#);
  }
  #[test]
  fn greater_2() {
    assert_case(
      r#"$CharacterEncoding; $CharacterEncoding = "ASCII"; a -> b"#,
      r#"a -> b"#,
    );
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"$CharacterEncoding; $CharacterEncoding = "ASCII"; a -> b; $CharacterEncoding = "UTF-8"; a -> b"#,
      r#"a -> b"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"$CharacterEncoding; $CharacterEncoding = "ASCII"; a -> b; $CharacterEncoding = "UTF-8"; a -> b; $CharacterEncoding = None; $SystemCharacterEncoding == $CharacterEncoding"#,
      r#""UTF-8" == None"#,
    );
  }
  #[test]
  fn system_character_encoding() {
    assert_case(r#"$SystemCharacterEncoding"#, r#""UTF-8""#);
  }
  #[test]
  fn root_directory() {
    assert_case(r#"$RootDirectory"#, r#""/""#);
  }
  #[test]
  fn input() {
    assert_case(r#"$Input"#, r#""""#);
  }
  #[test]
  fn pathname_separator() {
    assert_case(r#"$PathnameSeparator"#, r#""/""#);
  }
  #[test]
  fn export_formats() {
    assert_case(
      r#"$ExportFormats"#,
      r#"{"3DS", "AC", "ACO", "AIFF", "ArrowDataset", "ArrowIPC", "ASE", "AU", "AVI", "Base64", "Binary", "Bit", "BLEND", "BMP", "BREP", "BSON", "Byte", "BYU", "BZIP2", "C", "CDF", "CDXML", "Character16", "Character32", "Character8", "CIF", "CML", "Complex128", "Complex256", "Complex64", "CSV", "Cube", "CUR", "DAE", "DICOM", "DIF", "DIMACS", "DOT", "DTA", "DXF", "EPS", "ExpressionJSON", "ExpressionML", "FASTA", "FASTQ", "FBX", "FCS", "FITS", "FLAC", "FLV", "FMU", "GeoJSON", "GIF", "GLTF", "Graph6", "Graphlet", "GraphML", "GXL", "GZIP", "HarwellBoeing", "HDF", "HDF5", "HEIF", "HIN", "HTML", "HTMLFragment", "HTTPRequest", "HTTPResponse", "ICNS", "ICO", "IFC", "IGES", "Ini", "Integer128", "Integer16", "Integer24", "Integer32", "Integer64", "Integer8", "ISO", "JavaProperties", "JavaScriptExpression", "JPEG", "JPEG2000", "JSON", "JSONLD", "JSONLines", "JVX", "KML", "LEDA", "List", "LWO", "LXO", "Markdown", "MAT", "MathML", "Matroska", "Maya", "MCTT", "MGF", "MIDI", "MMCIF", "MMJSON", "MO", "MOL", "MOL2", "MP3", "MP4", "MS3D", "MTX", "MX", "MXNet", "NASACDF", "NB", "NetCDF", "NEXUS", "NOFF", "NQuads", "NTriples", "OBJ", "OFF", "Ogg", "ONNX", "OpenEXR", "ORC", "OWLFunctional", "Pajek", "Parquet", "PBM", "PCX", "PDB", "PDF", "PGM", "PHPIni", "PLY", "PNG", "PNM", "POR", "POV", "PPM", "PXR", "PythonExpression", "QuickTime", "RawBitmap", "RawJSON", "RDFXML", "Real128", "Real32", "Real64", "RIB", "RLE", "RTF", "SAS7BDAT", "SAV", "SCT", "SDF", "SMA", "SMILES", "SND", "SPARQLQuery", "SPARQLResultsJSON", "SPARQLResultsXML", "SPARQLUpdate", "Sparse6", "STEP", "STL", "String", "SurferGrid", "SVG", "Table", "TAR", "TerminatedString", "TeX", "TeXFragment", "Text", "TGA", "TGF", "TIFF", "TriG", "TSV", "Turtle", "UBJSON", "UnsignedInteger128", "UnsignedInteger16", "UnsignedInteger24", "UnsignedInteger32", "UnsignedInteger64", "UnsignedInteger8", "USD", "UUE", "VideoFrames", "VRML", "VTK", "WAV", "Wave64", "WDX", "WebP", "WL", "WLNet", "WMLF", "WXF", "X3D", "XBM", "XGL", "XHTML", "XHTMLMathML", "XLS", "XLSX", "XML", "XPORT", "XYZ", "ZIP", "ZPR", "ZSTD"}"#,
    );
  }
  #[test]
  fn import_formats() {
    assert_case(
      r#"$ImportFormats"#,
      r#"{"3DS", "7z", "AC", "ACO", "Affymetrix", "AgilentMicroarray", "AIFF", "ApacheLog", "ArcGRID", "ArrowDataset", "ArrowIPC", "ASC", "ASE", "AU", "AVI", "AVIF", "Base64", "BDF", "Binary", "BioImageFormat", "Bit", "BLEND", "BMP", "BREP", "BSON", "Byte", "BYU", "BZIP2", "CDED", "CDF", "CDX", "CDXML", "Character16", "Character32", "Character8", "CIF", "CML", "CommonLog", "Complex128", "Complex256", "Complex64", "CSV", "Cube", "CUR", "DAE", "DBF", "DICOM", "DICOMDIR", "DIF", "DIMACS", "Directory", "DOCX", "DOT", "DTA", "DXF", "EDF", "EML", "EPS", "ExpressionJSON", "ExpressionML", "ExtendedLog", "FASTA", "FASTQ", "FBX", "FCHK", "FCS", "FITS", "FLAC", "FLV", "GaussianLog", "GenBank", "GeoJSON", "GeoTIFF", "GGUF", "GIF", "GLTF", "GML", "GPX", "Graph6", "Graphlet", "GraphML", "GRIB", "GTOPO30", "GXF", "GXL", "GZIP", "HarwellBoeing", "HDF", "HDF5", "HEIF", "HIN", "HTML", "HTTPRequest", "HTTPResponse", "ICC", "ICNS", "ICO", "ICS", "IFC", "IGES", "Ini", "Integer128", "Integer16", "Integer24", "Integer32", "Integer64", "Integer8", "ISO", "JavaProperties", "JavaScriptExpression", "JCAMP-DX", "JPEG", "JPEG2000", "JSON", "JSONLD", "JSONLines", "JVX", "KML", "LaTeX", "LEDA", "List", "LWO", "LXO", "Markdown", "MAT", "MathML", "Matroska", "MBOX", "MCTT", "MDB", "MESH", "MGF", "MIDI", "MMCIF", "MMJSON", "MO", "MOBI", "MOL", "MOL2", "MP3", "MP4", "MPS", "MS3D", "MTP", "MTX", "MX", "MXNet", "NASACDF", "NB", "NDK", "NetCDF", "NEXUS", "NOFF", "NQuads", "NTriples", "OBJ", "ODS", "OFF", "Ogg", "ONNX", "OpenEXR", "ORC", "OSM", "OWLFunctional", "Pajek", "Parquet", "PBM", "PCAP", "PCX", "PDB", "PDF", "PEM", "PGM", "PHPIni", "PLY", "PNG", "PNM", "POR", "PPM", "PXR", "PythonExpression", "QuickTime", "RAR", "Raw", "RawBitmap", "RawJSON", "RData", "RDFXML", "RDS", "Real128", "Real32", "Real64", "RIB", "RLE", "RSS", "RTF", "SAS7BDAT", "SAV", "SCT", "SDF", "SDTS", "SDTSDEM", "SFF", "SHP", "SMA", "SME", "SMILES", "SND", "SP3", "SPARQLQuery", "SPARQLResultsJSON", "SPARQLResultsXML", "SPARQLUpdate", "Sparse6", "STEP", "STL", "String", "SurferGrid", "SVG", "SXC", "Table", "TAR", "TerminatedString", "TeX", "Text", "TGA", "TGF", "TIFF", "TIGER", "TLE", "TopoJSON", "TriG", "TSV", "Turtle", "UBJSON", "UnsignedInteger128", "UnsignedInteger16", "UnsignedInteger24", "UnsignedInteger32", "UnsignedInteger64", "UnsignedInteger8", "USD", "USGSDEM", "UUE", "VCF", "VCS", "VideoFormat", "VTK", "WARC", "WAV", "Wave64", "WDX", "WebP", "WL", "WLNet", "WMLF", "WXF", "X3D", "XBM", "XGL", "XHTML", "XHTMLMathML", "XLS", "XLSX", "XML", "XPORT", "XYZ", "ZIP", "ZSTD"}"#,
    );
  }
  #[test]
  fn machine_precision_2() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b; $MachinePrecision"#,
      r#"15.954589770191003"#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"ByteOrdering; ByteOrdering == -1 || ByteOrdering == 1; $ByteOrdering == ByteOrdering"#,
      r#"-1 == ByteOrdering"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(
      r#"ByteOrdering; ByteOrdering == -1 || ByteOrdering == 1; $ByteOrdering == ByteOrdering; $ByteOrdering == -1 || $ByteOrdering == 1"#,
      r#"True"#,
    );
  }
}
