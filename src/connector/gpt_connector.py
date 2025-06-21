import re
import time
import json
import openai
from abc import ABC, abstractmethod
from typing import AnyStr, List
from openai import RateLimitError, APIError
from openai import OpenAI

from .base_connector import Connector
from cdr import MessageRole, Message
from utils import count_tokens_w_msg, report_warning

class GPTConnector(Connector):
    def __init__(self, model_name='gpt-3.5-turbo', model_api=''):
        assert 'gpt' in model_name, "Model name must include 'gpt'"
        super().__init__(model_name, model_api)
        self.temperature = 0
        self.max_tokens = 800
        ## For SA since it returns rankedl list of headers
        # self.max_tokens = 3000
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.response_format = {"type": "json_object"}
        self.client = OpenAI(api_key=model_api) 

        assert model_api, "Model API key must be provided"
        openai.api_key = model_api



    def set_params(self, **params):
        self.temperature = params.get('temperature', self.temperature)
        self.max_tokens = params.get('max_tokens', self.max_tokens)
        self.top_p = params.get('top_p', self.top_p)
        self.frequency_penalty = params.get('frequency_penalty', self.frequency_penalty)
        self.presence_penalty = params.get('presence_penalty', self.presence_penalty)
        self.response_format = params.get('response_format', self.response_format)


    def _get_json(self, text):
        json_pattern = re.compile(r'\{[\s\S]*?\}')
        match = json_pattern.search(text)

        if match:
            json_str = match.group()
            try:
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError as e:
                print('json text:')
                print(text)
                raise ValueError(f"Failed to decode JSON: {e}")
        else:
            print('json text:')
            print(text)
            raise ValueError("No JSON object found in the string")
    
    def submit(self, msgs: List[Message], count_tokens: bool = False, verbose: bool = False, retry: int = 3,
               **kwargs):

        messages_builder = []
        for m in msgs:
            messages_builder.append({'role': m.role.value, 'content': m.content})

        if verbose:
            print('***GPT Request***')
            print(f"Request: {messages_builder}") 
            print('******************')

        num_retry = 0
        while num_retry < retry:
            try:
                response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages_builder,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        # top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty
                    )

                ans = response.choices[0].message.content
                json_ans = self._get_json(ans)
                break
            except RateLimitError as e:
                num_retry += 1
                report_warning(f"***Repeat for the {num_retry} times for RateLimitError***")
                print(e)
                time.sleep(20)
                continue
            except json.decoder.JSONDecodeError:
                num_retry += 1
                report_warning(f"***Repeat for the {num_retry} times for JSONDecodeError***")
                time.sleep(1)
                continue
            except APIError as e:
                # TODO: API error typically cannot be resolved.
                report_warning(f"***Repeat for the {num_retry} times for APIError***")
                report_warning(f"APIError: {e}")
                return {'response': {}, 'token_count': None}
            except Exception as e:
                num_retry += 1
                report_warning(f"***Repeat for the {num_retry} times for exception: {e}***")
                time.sleep(1)
                continue
        
        if num_retry == retry:
            return {'response': {}, 'token_count': None}

        if verbose:
            print('***GPT Response***')
            print(f"Response: {json_ans}")
            print('********************')

        if count_tokens:
            token_count = count_tokens_w_msg(messages_builder, model=self.model_name)
            return {'response': json_ans, 'token_count': token_count}
        else:
            return {'response': json_ans, 'token_count': None}        
    
    def __str__(self):
        return (f"GPTConnector(model_name='{self.model_name}', "
                f"temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, "
                f"top_p={self.top_p}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"presence_penalty={self.presence_penalty})")
