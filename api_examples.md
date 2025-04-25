# API Usage Examples

This document provides examples of how to call the GGUF Model Server API in various programming languages.

## Python Example

```python
import requests

api_url = "http://localhost:8000/generate"
api_key = "your_secret_api_key"  # Replace with your actual API key

headers = {
    "X-API-Key": api_key
}

data = {
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
}

response = requests.post(api_url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(result["generated_text"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## JavaScript/Node.js Example

```javascript
const fetch = require("node-fetch");

const apiUrl = "http://localhost:8000/generate";
const apiKey = "your_secret_api_key"; // Replace with your actual API key

async function generateText() {
	const response = await fetch(apiUrl, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"X-API-Key": apiKey,
		},
		body: JSON.stringify({
			prompt: "Once upon a time",
			max_tokens: 100,
			temperature: 0.7,
		}),
	});

	if (response.ok) {
		const result = await response.json();
		console.log(result.generated_text);
	} else {
		console.error(`Error: ${response.status}`);
		console.error(await response.text());
	}
}

generateText();
```

## cURL Example

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Java Example

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class ModelApiExample {
    public static void main(String[] args) {
        try {
            String apiUrl = "http://localhost:8000/generate";
            String apiKey = "your_secret_api_key";  // Replace with your actual API key

            String requestBody = """
                {
                    "prompt": "Once upon a time",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                """;

            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl))
                .header("Content-Type", "application/json")
                .header("X-API-Key", apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                System.out.println(response.body());
            } else {
                System.out.println("Error: " + response.statusCode());
                System.out.println(response.body());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Go Example

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

type Request struct {
	Prompt           string  `json:"prompt"`
	MaxTokens        int     `json:"max_tokens"`
	Temperature      float64 `json:"temperature"`
}

type Response struct {
	GeneratedText string `json:"generated_text"`
}

func main() {
	apiURL := "http://localhost:8000/generate"
	apiKey := "your_secret_api_key" // Replace with your actual API key

	requestData := Request{
		Prompt:      "Once upon a time",
		MaxTokens:   100,
		Temperature: 0.7,
	}

	requestBody, err := json.Marshal(requestData)
	if err != nil {
		fmt.Printf("Error marshaling request: %v\n", err)
		return
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(requestBody))
	if err != nil {
		fmt.Printf("Error creating request: %v\n", err)
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("Error making request: %v\n", err)
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Error reading response: %v\n", err)
		return
	}

	if resp.StatusCode == 200 {
		var response Response
		err = json.Unmarshal(body, &response)
		if err != nil {
			fmt.Printf("Error unmarshaling response: %v\n", err)
			return
		}
		fmt.Println(response.GeneratedText)
	} else {
		fmt.Printf("Error: %d\n", resp.StatusCode)
		fmt.Println(string(body))
	}
}
```
