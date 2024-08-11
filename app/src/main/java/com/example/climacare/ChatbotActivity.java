package com.example.climacare;

import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class ChatbotActivity extends BaseActivity {

    private static final String TAG = "ChatbotActivity";

    private Spinner optionsSpinner;
    private Button sendChatButton;
    private TextView chatResponseTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.chatbot_layout);

        optionsSpinner = findViewById(R.id.optionsSpinner);
        sendChatButton = findViewById(R.id.sendChatButton);
        chatResponseTextView = findViewById(R.id.chatResponseTextView);

        // Load the chatbot responses from the file and populate the spinner
        new LoadChatbotResponsesTask().execute();

        // Set a listener for the send button
        sendChatButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String selectedOption = optionsSpinner.getSelectedItem().toString();
                if (!selectedOption.equals("Select an option below")) {
                    new ChatbotTask().execute(selectedOption);
                } else {
                    chatResponseTextView.setText("Please select an option.");
                }
            }
        });
    }

    private class LoadChatbotResponsesTask extends AsyncTask<Void, Void, List<String>> {
        @Override
        protected List<String> doInBackground(Void... voids) {
            List<String> patterns = new ArrayList<>();
            patterns.add("Select an option below");
            try {
                InputStream inputStream = getAssets().open("chatbot_responses.txt");
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\\|");
                    if (parts.length == 2) {
                        patterns.add(parts[0].trim());
                        Log.d(TAG, "Added pattern: " + parts[0].trim());
                    } else {
                        Log.w(TAG, "Skipping malformed line: " + line);
                    }
                }
                reader.close();
            } catch (IOException e) {
                Log.e(TAG, "Error reading chatbot responses file: " + e.getMessage());
            }
            return patterns;
        }

        @Override
        protected void onPostExecute(List<String> patterns) {
            super.onPostExecute(patterns);
            if (patterns != null && !patterns.isEmpty()) {
                ArrayAdapter<String> adapter = new ArrayAdapter<>(ChatbotActivity.this, R.layout.spinner_item, patterns);
                adapter.setDropDownViewResource(R.layout.spinner_item);
                optionsSpinner.setAdapter(adapter);
            } else {
                Log.e(TAG, "No patterns found in the chatbot responses file.");
            }
        }
    }

    private class ChatbotTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            try {
                String apiUrl = "http://192.168.150.54:5000/chat";
                URL url = new URL(apiUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json");
                connection.setDoOutput(true);

                JSONObject postData = new JSONObject();
                postData.put("message", params[0]);

                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());
                outputStream.write(postData.toString().getBytes());
                outputStream.flush();
                outputStream.close();

                InputStream inputStream = connection.getInputStream();
                InputStreamReader reader = new InputStreamReader(inputStream);
                StringBuilder response = new StringBuilder();
                char[] buffer = new char[1024]; // Buffer size can be adjusted
                int bytesRead;
                while ((bytesRead = reader.read(buffer)) != -1) {
                    response.append(buffer, 0, bytesRead);
                }

                inputStream.close();
                connection.disconnect();

                return response.toString();
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Error: " + e.getMessage());
                return null;
            }
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            // Show loading indicator
            chatResponseTextView.setText("Loading...");
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            // Hide loading indicator and handle the response
            if (result != null) {
                try {
                    JSONObject jsonResponse = new JSONObject(result);
                    String chatbotResponse = jsonResponse.getString("response");
                    chatResponseTextView.setText(chatbotResponse); // Display the entire response at once
                } catch (JSONException e) {
                    Log.e(TAG, "Error parsing JSON: " + e.getMessage());
                    chatResponseTextView.setText("Error: Failed to parse server response.");
                }
            } else {
                Log.e(TAG, "Null response received from server.");
                chatResponseTextView.setText("Null response received from server.");
            }
        }
    }
}
