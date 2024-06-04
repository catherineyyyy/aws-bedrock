import boto3
import json
import os

bedrock_runtime = boto3.client('bedrock-runtime')

prompt = "what is the capital of india"

kwargs = {
  "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
  "contentType": "application/json",
  "accept": "application/json",
  "body": json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          }
        ]
      }
    ]
  })
}

response = bedrock_runtime.invoke_model(**kwargs)

body = json.loads(response['body'].read())
print(body)