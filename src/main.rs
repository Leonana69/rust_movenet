use std::slice::from_raw_parts;
use opencv::core::{flip};
use opencv::videoio::*;
use opencv::{
	prelude::*,
	videoio,
	highgui::*,
};

mod utils;
use utils::*;

fn main() {
	// load model and create interpreter
	let mut interpreter = load_model("resource/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite");
	let input_index = interpreter.inputs().to_vec()[0];
	let output_heatmap_index = interpreter.outputs().to_vec()[0];
	let output_offset_index = interpreter.outputs().to_vec()[1];
	
	// open camera
	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap(); // 0 is the default camera
	videoio::VideoCapture::is_opened(&cam).expect("Open camera [FAILED]");
	cam.set(CAP_PROP_FPS, 30.0).expect("Set camera FPS [FAILED]");

	loop {
		let mut frame = Mat::default();
		cam.read(&mut frame).expect("VideoCapture: read [FAILED]");

		if frame.size().unwrap().width > 0 {
			// flip the image horizontally
			let mut flipped = Mat::default();
			flip(&frame, &mut flipped, 1).expect("flip [FAILED]");

			let resized_img = resize_with_padding(&flipped, [257, 257]);

			// get slice of buffer for fast copy
			let input_buffer = interpreter.tensor_data_mut::<f32>(input_index).unwrap();
			let data_buffer = unsafe { from_raw_parts(resized_img.data(), 257 * 257 * 3) };
			for i in 0..257 * 257 * 3 {
				// normalize pixel value to [-0.5, 0.5]
				input_buffer[i] = data_buffer[i] as f32 / 127.5 - 1.0;
			}
			// the interpreter is quite slow
			interpreter.invoke().expect("Invoke [FAILED]");

			// get output
			let output_heatmap: &[f32] = interpreter.tensor_data(output_heatmap_index).unwrap();
			let output_offset: &[f32] = interpreter.tensor_data(output_offset_index).unwrap();
			draw_keypoints(&mut flipped, output_heatmap, output_offset, 0.25);
			imshow("PoseNet", &flipped).expect("imshow [ERROR]");
		}
		// keypress check
		let key = wait_key(1).unwrap();
		if key > 0 && key != 255 {
			break;
		}
	}
}