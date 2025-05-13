import functools, torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info

class LiveCCDemoInfer:
  fps = 2
  initial_fps_frames = 6
  streaming_fps_frames = 2
  initial_time_interval = initial_fps_frames / fps
  streaming_time_interval = streaming_fps_frames / fps
  frame_time_interval = 1 / fps

  def __init__(self, model_path: str = None, device: str = 'cuda'):
      self.model = Qwen2VLForConditionalGeneration.from_pretrained(
          model_path, torch_dtype="auto",
          device_map=device,
          attn_implementation='flash_attention_2'
      )
      self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
      self.streaming_eos_token_id = self.processor.tokenizer(' ...').input_ids[-1]
      self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
      message = {
          "role": "user",
          "content": [
              {"type": "text", "text": 'livecc'},
          ]
      }
      texts = self.processor.apply_chat_template([message], tokenize=False)
      self.system_prompt_offset = texts.index('<|im_start|>user')

  def video_qa(
      self,
      message: str,
      state: dict,
      do_sample: bool = True,
      repetition_penalty: float = 1.05,
      **kwargs,
  ):
      """
      state: dict, (maybe) with keys:
          video_path: str, video path
          video_timestamp: float, current video timestamp
          last_timestamp: float, last processed video timestamp
          last_video_pts_index: int, last processed video frame index
          video_pts: np.ndarray, video pts
          last_history: list, last processed history
          past_key_values: llm past_key_values
          past_ids: past generated ids
      """
      video_path = state.get('video_path', None)
      conversation = []
      past_ids = state.get('past_ids', None)
      content = [{"type": "text", "text": message}]
      if past_ids is None and video_path: # only use once
          content.insert(0, {"type": "video", "video": video_path})
      conversation.append({"role": "user", "content": content})
      image_inputs, video_inputs = process_vision_info(conversation)
      texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
      if past_ids is not None:
          texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
      inputs = self.processor(
          text=texts,
          images=image_inputs,
          videos=video_inputs,
          return_tensors="pt",
          return_attention_mask=False
      )
      inputs.to(self.model.device)
      if past_ids is not None:
          inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1)
      outputs = self.model.generate(
          **inputs, past_key_values=state.get('past_key_values', None),
          return_dict_in_generate=True, do_sample=do_sample,
          repetition_penalty=repetition_penalty,
          max_new_tokens=512,
      )
      state['past_key_values'] = outputs.past_key_values
      state['past_ids'] = outputs.sequences[:, :-1]
      response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
      return response, state

model_path = 'chenjoya/LiveCC-7B-Instruct'
# download a test video at: https://github.com/showlab/livecc/blob/main/demo/sources/howto_fix_laptop_mute_1080p.mp4
video_path = "demo/sources/howto_fix_laptop_mute_1080p.mp4"

infer = LiveCCDemoInfer(model_path=model_path)
state = {'video_path': video_path}
# first round
query1 = 'What is the video?'
response1, state = infer.video_qa(message=query1, state=state)
print(f'Q1: {query1}\nA1: {response1}')
# second round
query2 = 'How do you know that?'
response2, state = infer.video_qa(message=query2, state=state)
print(f'Q2: {query2}\nA2: {response2}')
