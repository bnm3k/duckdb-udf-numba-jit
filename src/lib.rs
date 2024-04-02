#![allow(dead_code)]
#![allow(unused_imports)]

use arrow::array::Array;
use arrow::array::{self as arr, Float64Array};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use eyre::{Context, ContextCompat, Result};
use pyo3::prelude::*;

fn calc_haversine_dist(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    // x -> longitude
    // y -> latitude
    const EARTH_RADIUS: f64 = 6372.8; // km
    let radians = |d| (d * std::f64::consts::PI) / 180.0;
    let square = |x| x * x;

    let p0_latitude = radians(y0);
    let p1_latitude = radians(y1);

    let delta_latitude = (y0 - y1).to_radians();
    let delta_longitude = (x0 - x1).to_radians();

    let central_angle_inner = square((delta_latitude / 2.0).sin())
        + p0_latitude.cos() * p1_latitude.cos() * square((delta_longitude / 2.0).sin());
    let central_angle = 2.0 * central_angle_inner.sqrt().asin();

    let distance = EARTH_RADIUS * central_angle;
    return distance;
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn haversine_dist<'a>(
    py: Python,
    x0: &PyAny,
    y0: &PyAny,
    x1: &PyAny,
    y1: &PyAny,
) -> Result<Py<PyAny>> {
    let arr = |ns| -> PyResult<Float64Array> {
        let arraydata = arrow::array::ArrayData::from_pyarrow(ns)?;
        return Ok(Float64Array::from(arraydata));
    };

    // get string lengths
    let x0 = arr(x0)?;
    let y0 = arr(y0)?;
    let x1 = arr(x1)?;
    let y1 = arr(y1)?;
    // TODO handle nulls?
    let distances_arr = {
        let mut arr_builder = arr::Float64Builder::with_capacity(x0.len());
        for i in 0..x0.len() {
            let dist = unsafe {
                calc_haversine_dist(
                    x0.value_unchecked(i),
                    y0.value_unchecked(i),
                    x1.value_unchecked(i),
                    y1.value_unchecked(i),
                )
            };
            arr_builder.append_value(dist)
        }
        arr_builder.finish()
    };
    let output = distances_arr.to_data();

    output
        .to_pyarrow(py)
        .context("Could not convert to pyarrow")
}

/// A Python module implemented in Rust.
#[pymodule]
fn udfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine_dist, m)?)?;
    Ok(())
}
