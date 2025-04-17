# Authentication

## Getting auth token from Fireworks

First, you need to login with `firectl signin`. After signin, the file `~/.fireworks/auth.ini` will be populated. That file will have
- account_id
- id_token
- refresh_token

and `reward_kit` will use the id token to authenticate with the server

## Interacting with the Fireworks RESTFUL API
All requests made to the Fireworks AI via REST API must include an Authorization header.
Header should specify a valid Bearer Token with API key and must be encoded as JSON with the “Content-Type: application/json” header.
This ensures that your requests are properly authenticated and formatted for interaction with the Fireworks AI.
A Sample header to be included in the REST API request should look like below:

authorization: Bearer <auth_token>

## example code for list dataset with the api

```
import requests

url = "https://api.fireworks.ai/v1/accounts/{account_id}/datasets"

headers = {"Authorization": "Bearer <token>"}

response = requests.request("GET", url, headers=headers)

print(response.text)
```