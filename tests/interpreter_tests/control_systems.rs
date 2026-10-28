use super::*;

fn eval(code: &str) -> String {
  clear_state();
  interpret(code).unwrap()
}

mod transfer_function_model {
  use super::*;

  // A model is kept as `{nums, den}`: one shared denominator when every
  // element has it, a matrix of them otherwise; nothing is cancelled.
  #[test]
  fn normal_form() {
    for (input, expected) in [
      (
        "TransferFunctionModel[(s + 1)/(s^2 + 2 s + 3), s]",
        "TransferFunctionModel[{{{1 + s}}, 3 + 2*s + s^2}, s]",
      ),
      (
        "TransferFunctionModel[{{(s + 1)/(s + 2), 1/s}}, s]",
        "TransferFunctionModel[{{{1 + s, 1}}, {{2 + s, s}}}, s]",
      ),
      (
        "TransferFunctionModel[{{1/(s + 1), 1/(s + 1)}}, s]",
        "TransferFunctionModel[{{{1, 1}}, 1 + s}, s]",
      ),
      (
        "TransferFunctionModel[(s^2 - 1)/(s^2 + 3 s + 2), s]",
        "TransferFunctionModel[{{{-1 + s^2}}, 2 + 3*s + s^2}, s]",
      ),
      (
        "TransferFunctionModel[1/s + 1/(s + 1), s]",
        "TransferFunctionModel[{{{1 + 2*s}}, s*(1 + s)}, s]",
      ),
      (
        "TransferFunctionModel[{{1}}, s]",
        "TransferFunctionModel[{{{1}}, 1}, s]",
      ),
      (
        "TransferFunctionModel[{{1/(s + 1)}}, s, SamplingPeriod -> 0.1]",
        "TransferFunctionModel[{{{1}}, 1 + s}, s, SamplingPeriod -> 0.1]",
      ),
    ] {
      assert_eq!(eval(input), expected, "{input}");
    }
  }

  // A delay stays a factor of the numerator, written exactly.
  #[test]
  fn time_delays() {
    assert_eq!(
      eval(
        "ToString[TransferFunctionModel[{{10/(E^(2.*s)*(1 + 7.*s))}}, s], \
         InputForm]"
      ),
      "TransferFunctionModel[{{{10./E^(2*s)}}, 1. + 7.*s}, s]"
    );
    assert_eq!(
      eval("TransferFunctionModel[{{10/(E^(2.*s)*(1 + 7.*s))}}, s][1.]"),
      "{{1.25*SystemsModelDelay[2]}}"
    );
  }

  #[test]
  fn evaluation_poles_and_zeros() {
    assert_eq!(
      eval("TransferFunctionModel[(s + 1)/(s^2 + 2 s + 3), s][2]"),
      "{{3/11}}"
    );
    assert_eq!(
      eval(
        "TransferFunctionPoles[TransferFunctionModel[(s + 1)/(s^2 + 2 s + 3), s]]"
      ),
      "{{{-1 - I*Sqrt[2], -1 + I*Sqrt[2]}}}"
    );
    assert_eq!(
      eval(
        "TransferFunctionZeros[TransferFunctionModel[(s + 1)/(s^2 + 2 s + 3), s]]"
      ),
      "{{{-1}}}"
    );
    assert_eq!(
      eval(
        "TransferFunctionPoles[TransferFunctionModel[{{1/(s + 1), 1/(s + 2)}}, s]]"
      ),
      "{{{-1}, {-2}}}"
    );
  }
}

mod output_response {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      eval("OutputResponse[TransferFunctionModel[1/(s + 1), s], 1, t]"),
      "{(-1 + E^t)/E^t}"
    );
    assert_eq!(
      eval(
        "OutputResponse[TransferFunctionModel[1/(s + 1), s], DiracDelta[t], t]"
      ),
      "{HeavisideTheta[t]/E^t}"
    );
    // A delay cannot be simulated exactly.
    clear_state();
    let r = interpret_with_stdout(
      "OutputResponse[TransferFunctionModel[Exp[-2 s]/(s + 1), s], 1, t]",
    )
    .unwrap();
    assert_eq!(
      r.result,
      "OutputResponse[TransferFunctionModel[{{{E^(-2*s)}}, 1 + s}, s], 1, t]"
    );
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("OutputResponse::delaysm")),
      "{:?}",
      r.warnings
    );
  }

  // Over a time span the response is simulated: `{InterpolatingFunction[…][t]}`.
  #[test]
  fn simulated() {
    for (input, expected) in [
      (
        "r = OutputResponse[TransferFunctionModel[1/(s + 1), s], 1, {t, 0, 10}]; \
         Round[Table[r[[1]] /. t -> x, {x, {0., 1., 2.5, 10.}}], 10.^-5]",
        "{0., 0.63212, 0.9179200000000001, 0.9999500000000001}",
      ),
      // The delay holds the response back.
      (
        "r = OutputResponse[TransferFunctionModel[{{10/(E^(2.*s)*(1 + 7.*s))}}, s], \
         1, {t, 0, 100}]; Round[Table[r[[1]] /. t -> x, {x, {1., 5., 30.}}], 10.^-3]",
        "{0., 3.486, 9.817}",
      ),
      (
        "r = OutputResponse[TransferFunctionModel[(0.165 + 0.85*s)/(0.165 + 0.85*s + \
         1.*s^2), s], 0.1 Sin[2 Pi 0.5 t], {t, 0, 20}]; \
         Round[Table[r[[1]] /. t -> x, {x, {1., 5., 13.3}}], 10.^-4]",
        "{0.0392, 0.025900000000000003, 0.010700000000000001}",
      ),
      (
        "Head[OutputResponse[TransferFunctionModel[1/(s + 1), s], 1, {t, 0, 10}][[1, 0]]]",
        "InterpolatingFunction",
      ),
    ] {
      assert_eq!(eval(input), expected, "{input}");
    }
  }
}

mod frequency_response_plots {
  use super::*;

  #[test]
  fn bode_nyquist_and_nichols() {
    assert_eq!(
      eval(
        "b = BodePlot[TransferFunctionModel[1/(s^2 + 2 s + 1), s]]; \
         {Head[b], Dimensions[b[[1]]]}"
      ),
      "{Grid, {2, 1}}"
    );
    assert_eq!(
      eval("Head[NyquistPlot[TransferFunctionModel[1/(s^2 + 2 s + 1), s]]]"),
      "Graphics"
    );
    assert_eq!(
      eval("Head[NicholsPlot[TransferFunctionModel[Exp[-s]/(s + 1), s]]]"),
      "Graphics"
    );
    // A list of models and an explicit frequency range.
    assert_eq!(
      eval(
        "Head[BodePlot[{TransferFunctionModel[1/(s + 1), s], \
         TransferFunctionModel[1/(s + 2), s]}, {0.1, 100}]]"
      ),
      "Grid"
    );
  }
}

mod discrete_time {
  use super::*;

  // The bilinear transform (the default) and the rectangular rules, both
  // ways; the forms differ from wolframscript's in where `1.` appears.
  #[test]
  fn conversions() {
    assert_eq!(
      eval(
        "m = ToDiscreteTimeModel[TransferFunctionModel[1/(s^2 + 3 s + 2), s], \
         0.5, z]; {m[[1, 2, 1, 1]], m[[3]]}"
      ),
      "{1.5 - 7.*z + 7.5*z^2, SamplingPeriod -> 0.5}"
    );
    assert_eq!(
      eval(
        "ToDiscreteTimeModel[TransferFunctionModel[1/(s + 1), s], 0.1, z, \
         Method -> \"BackwardRectangularRule\"]"
      ),
      "TransferFunctionModel[{{{0.1*z}}, {{-1 + 1.1*z}}}, z, SamplingPeriod -> 0.1]"
    );
    assert_eq!(
      eval(
        "m = ToContinuousTimeModel[TransferFunctionModel[(1.*z)/(0.68 - 0.4*z + \
         z^2), z, SamplingPeriod -> 1.], s]; \
         {m[[1, 1, 1, 1]], Round[CoefficientList[m[[1, 2, 1, 1]], s], 10.^-10], \
         m[[3]]}"
      ),
      "{1. - 0.25*s^2, {1.28, 0.32, 0.52}, SamplingPeriod -> None}"
    );
  }

  // A sampled model's response over {n, nmin, nmax} (in time units, one
  // value per period) or to a list of samples.
  #[test]
  fn responses() {
    let tf = "tf = TransferFunctionModel[(1.*z)/(0.68 - 0.4*z + z^2), z, \
              SamplingPeriod -> 1.]; ";
    assert_eq!(
      eval(&format!(
        "{tf}Round[OutputResponse[tf, DiscreteDelta[n], {{n, 0, 4}}], 10.^-10]"
      )),
      "{{0., 1., 0.4, -0.52, -0.48000000000000004}}"
    );
    assert_eq!(
      eval(&format!("{tf}OutputResponse[tf, {{0, 1, 2, 3, 4}}]")),
      "{{0., 0., 1., 2.4, 3.28}}"
    );
    assert_eq!(
      eval(
        "Length[First[OutputResponse[TransferFunctionModel[(z + 0.5)/(z^2 - \
         0.5), z, SamplingPeriod -> 0.1], 1, {n, 0, 4}]]]"
      ),
      "41"
    );
    // A continuous impulse acts through the state.
    assert_eq!(
      eval(
        "Round[OutputResponse[TransferFunctionModel[1/(s + 1), s], DiracDelta[t], \
         {t, 0, 2}][[1]] /. t -> 1., 10.^-6]"
      ),
      "0.36787899999999996"
    );
    assert_eq!(
      eval(
        "Head[BodePlot[TransferFunctionModel[(1.*z)/(0.68 - 0.4*z + z^2), z, \
         SamplingPeriod -> 1.], {0.001, 2 Pi - 0.0001}]]"
      ),
      "Grid"
    );
  }
}
