package com.example.climacare;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ObjectAnimator;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends BaseActivity {

    private static final String TAG = "MainActivity";

    private EditText locationEditText;
    private Button predictButton;
    private TextView secondaryTextView;
    private WeatherDbManager dbManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        locationEditText = findViewById(R.id.locationEditText);
        predictButton = findViewById(R.id.predictButton);

        dbManager = new WeatherDbManager(this);
        dbManager.open();

        // Get the username from intent extras
        String username = getIntent().getStringExtra("username");
        if (username == null) {
            // If username is not passed, retrieve from SharedPreferences
            SharedPreferences prefs = getSharedPreferences("MyPrefs", MODE_PRIVATE);
            username = prefs.getString("username", "User");
        } else {
            // Save the username to SharedPreferences for future use
            SharedPreferences.Editor editor = getSharedPreferences("MyPrefs", MODE_PRIVATE).edit();
            editor.putString("username", username);
            editor.apply();
        }

        if (username != null) {
            // Set the welcome message in the header
            TextView headerTextView = findViewById(R.id.headerTextView);
            headerTextView.setText("Welcome " + username + ",");

            // Show the welcome message for 3 seconds
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    fadeOutAndIn(headerTextView, "To continue, enter the location below");
                }
            }, 3000);
        }

        // Initialize secondaryTextView
        secondaryTextView = findViewById(R.id.secondaryTextView);

        // Show the instruction message for 5 seconds and then hide it
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                secondaryTextView.setVisibility(View.GONE);
            }
        }, 5000);

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String location = locationEditText.getText().toString();
                new WeatherPredictionTask().execute(location);
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        dbManager.close();
    }

    // Method to create a smooth fade out and fade in effect
    private void fadeOutAndIn(final TextView textView, final String newText) {
        ObjectAnimator fadeOut = ObjectAnimator.ofFloat(textView, "alpha", 1f, 0f);
        fadeOut.setDuration(1000); // Duration of fade out

        ObjectAnimator fadeIn = ObjectAnimator.ofFloat(textView, "alpha", 0f, 1f);
        fadeIn.setDuration(1000); // Duration of fade in

        fadeOut.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                textView.setText(newText);
                fadeIn.start();
            }
        });

        fadeOut.start();
    }

    private class WeatherPredictionTask extends AsyncTask<String, Void, String> {

        private String location; // Declare location variable

        @Override
        protected String doInBackground(String... params) {
            String apiUrl = "http://192.168.150.54:5000/predict";
            String location = params[0];
            this.location = params[0]; // Assign value to class-level variable

            try {
                URL url = new URL(apiUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json");
                connection.setDoOutput(true);

                JSONObject postData = new JSONObject();
                postData.put("city", location);

                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());
                outputStream.write(postData.toString().getBytes());
                outputStream.flush();
                outputStream.close();

                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    InputStream inputStream = connection.getInputStream();
                    InputStreamReader reader = new InputStreamReader(inputStream);
                    StringBuilder response = new StringBuilder();
                    int data = reader.read();
                    while (data != -1) {
                        char current = (char) data;
                        response.append(current);
                        data = reader.read();
                    }
                    return response.toString();
                } else {
                    return null;
                }
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Error: " + e.getMessage());
                return null;
            }

        }

        @Override
        protected void onPostExecute(String result) {
            if (result != null) {
                try {
                    JSONObject jsonObject = new JSONObject(result);
                    String prediction = jsonObject.getString("prediction");
                    JSONArray dailyForecasts = jsonObject.getJSONArray("daily_forecasts");

                    double meanTemperature = 0;
                    double meanPrecipitation = 0;

                    for (int i = 0; i < dailyForecasts.length(); i++) {
                        JSONObject dayForecast = dailyForecasts.getJSONObject(i);
                        meanTemperature += dayForecast.getDouble("predicted_temperature");
                        meanPrecipitation += dayForecast.getDouble("predicted_precipitation");
                    }

                    meanTemperature /= dailyForecasts.length();
                    meanPrecipitation /= dailyForecasts.length();

                    // Save prediction to the database
                    dbManager.addPrediction(location, prediction, meanTemperature, meanPrecipitation);

                    // Show the prediction result in a popup window
                    showPredictionPopup(location, meanTemperature, meanPrecipitation, prediction);

                } catch (JSONException e) {
                    Log.e(TAG, "Error: " + e.getMessage());
                }
            } else {
                Toast.makeText(MainActivity.this, "Failed to get prediction. Please try again.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Method to show the prediction result in a popup window
    private void showPredictionPopup(String location, double meanTemperature, double meanPrecipitation, String prediction) {
        View popupView = getLayoutInflater().inflate(R.layout.prediction_popup, null);
        TextView locationTextView = popupView.findViewById(R.id.locationTextView);
        TextView predictionResultTextView = popupView.findViewById(R.id.predictionResultTextView);
        TextView meanTemperatureTextView = popupView.findViewById(R.id.meanTemperatureTextView);
        TextView meanPrecipitationTextView = popupView.findViewById(R.id.meanPrecipitationTextView);

        // Format the values to two decimal places
        String meanTemperatureStr = String.format("%.2f", meanTemperature);
        String meanPrecipitationStr = String.format("%.2f", meanPrecipitation);

        locationTextView.setText("Location: " + location);
        predictionResultTextView.setText("Prediction: " + prediction);
        meanTemperatureTextView.setText("Mean Temperature next 7 days: " + meanTemperatureStr + "°C");
        meanPrecipitationTextView.setText("Mean Precipitation next 7 days: " + meanPrecipitationStr + "mm");

        final PopupWindow popupWindow = new PopupWindow(popupView, WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT);
        popupWindow.setFocusable(true);

        // Show the popup window at the center of the screen
        popupWindow.showAtLocation(findViewById(android.R.id.content), Gravity.CENTER, 0, 340);

        // Close the popup window after 8 seconds
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                popupWindow.dismiss();
            }
        }, 8000);
    }
}
