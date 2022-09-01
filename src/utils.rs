use opencv::{
	prelude::*,
	imgproc::*,
	core::*,
	highgui::*,
};

use tflite::ops::builtin::*;
use tflite::{FlatBufferModel, InterpreterBuilder, Interpreter};

pub fn resize_with_padding(img: &Mat, new_shape: [i32;2]) -> Mat {
	let img_shape = [img.cols(), img.rows()];
	let width: i32;
	let height: i32;
	if img_shape[0] as f64 / img_shape[1] as f64 > new_shape[0] as f64 / new_shape[1] as f64 {
		width = new_shape[0];
		height = (new_shape[0] as f64 / img_shape[0] as f64 * img_shape[1] as f64) as i32;
	} else {
		width = (new_shape[1] as f64 / img_shape[1] as f64 * img_shape[0] as f64) as i32;
		height = new_shape[1];
	}

	let mut resized = Mat::default();
	resize(
		img,
		&mut resized,
		Size { width, height },
		0.0, 0.0,
		INTER_LINEAR)
		.expect("resize_with_padding: resize [FAILED]");

	let delta_w = new_shape[0] - width;
	let delta_h = new_shape[1] - height;
	let (top, bottom) = (delta_h / 2, delta_h - delta_h / 2);
	let (left, right) = (delta_w / 2, delta_w - delta_w / 2);
		
	let mut rslt = Mat::default();
	copy_make_border(
		&resized,
		&mut rslt,
		top, bottom, left, right,
		BORDER_CONSTANT,
		Scalar::new(0.0, 0.0, 0.0, 0.0))
		.expect("resize_with_padding: copy_make_border [FAILED]");
	rslt
}

fn parse_keypoints(heatmap: &[f32], offset: &[f32], threshold: f32) -> Vec<Vec<f32>> {
	let mut pose_kpts = vec![vec![0.0 as f32; 3]; 17];
	for index in 0..17 {
		let mut max_val = 0.0 as f32;
		let mut max_row = 0;
		let mut max_col = 0;
		for r in 0..9 {
			for c in 0..9 {
				if heatmap[r * 9 * 17 + c * 17 + index] > max_val {
					max_val = heatmap[r * 9 * 17 + c * 17 + index];
					max_row = r;
					max_col = c;
				}
			}
		}

		pose_kpts[index][0] = max_col as f32 / 8.0 * 257.0 + offset[max_row * 9 * 34 + max_col * 34 + index + 17];
		pose_kpts[index][1] = max_row as f32 / 8.0 * 257.0 + offset[max_row * 9 * 34 + max_col * 34 + index];
		if max_val > threshold {
			pose_kpts[index][2] = 1.0;
		}
	}
	pose_kpts
}

pub fn draw_keypoints(img: &mut Mat, heatmap: &[f32], offset: &[f32], threshold: f32) {
	// heatmap: [9, 9, 17]
	// offset: [9, 9, 34]
	let pose_kpts = parse_keypoints(heatmap, offset, threshold);

	let ratio: f32;
	let pad_x: i32;
	let pad_y: i32;
	if img.rows() > img.cols() {
		ratio = img.rows() as f32 / 257.0;
		pad_x = (img.rows() - img.cols()) / 2;
		pad_y = 0;
	} else {
		ratio = img.cols() as f32 / 257.0;
		pad_x = 0;
		pad_y = (img.cols() - img.rows()) / 2;
	}

	for index in 0..17 {
		if pose_kpts[index][2] > 0.0 {
			circle(img,
				Point { x: (ratio * pose_kpts[index][0]) as i32 - pad_x, y: (ratio * pose_kpts[index][1]) as i32 - pad_y},
				0,
				Scalar::new(0.0, 255.0, 0.0, 0.0),
				5, LINE_AA, 0).expect("Draw circle [FAILED]");
		}
	}
}

pub fn load_model(file_name: &str) -> Interpreter<BuiltinOpResolver> {
	let model = FlatBufferModel::build_from_file(
		file_name)
		.expect("FlatBufferModel: load file [FAILED]");

	let builder = InterpreterBuilder::new(
		model,
		BuiltinOpResolver::default()
	).expect("InterpreterBuilder init [FAILED]");

	let mut interpreter = builder.build().expect("Build interpreter [FAILED]");
	interpreter.allocate_tensors().expect("Tensor allocation [FAILED]");
	interpreter
}