package com.example.climacare;

import android.app.AlertDialog;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;

public class forecast extends BaseActivity {

    private EditText editTextCity;
    private Button buttonSend;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.forecast);

        editTextCity = findViewById(R.id.editTextCity);
        buttonSend = findViewById(R.id.buttonSend);

        buttonSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String city = editTextCity.getText().toString().trim();
                if (!city.isEmpty()) {
                    new FetchWeatherForecastTask().execute(city);
                } else {
                    Toast.makeText(forecast.this, "Please enter a city name", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private class FetchWeatherForecastTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            String city = params[0];
            String urlString = "http://192.168.150.54:5000/predict";

            try {
                URL url = new URL(urlString);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setRequestProperty("Content-Type", "application/json; utf-8");
                connection.setRequestProperty("Accept", "application/json");
                connection.setDoOutput(true);

                String jsonInputString = "{\"city\": \"" + city + "\"}";
                try (java.io.OutputStream os = connection.getOutputStream()) {
                    byte[] input = jsonInputString.getBytes("utf-8");
                    os.write(input, 0, input.length);
                }

                BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream(), "utf-8"));
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                return response.toString();
            } catch (Exception e) {
                e.printStackTrace();
                return null; // Return null in case of error
            }
        }

        @Override
        protected void onPostExecute(String result) {
            if (result != null) {
                try {
                    JSONObject jsonObject = new JSONObject(result);
                    String city = jsonObject.getString("city");
                    JSONArray dailyForecasts = jsonObject.getJSONArray("daily_forecasts");

                    StringBuilder forecast = new StringBuilder();
                    forecast.append("City: ").append(city).append("\n\n");

                    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
                    SimpleDateFormat dayFormat = new SimpleDateFormat("EEEE", Locale.getDefault());

                    for (int i = 0; i < dailyForecasts.length(); i++) {
                        JSONObject dayForecast = dailyForecasts.getJSONObject(i);
                        String dateStr = dayForecast.getString("date");
                        Date date = dateFormat.parse(dateStr);
                        String dayOfWeek = dayFormat.format(date);

                        double predictedTemperature = dayForecast.getDouble("predicted_temperature");
                        double predictedPrecipitation = dayForecast.getDouble("predicted_precipitation");

                        forecast.append("Date: ").append(dateStr).append(" (").append(dayOfWeek).append(")\n")
                                .append("Predicted Temperature: ").append(predictedTemperature).append(" °C\n")
                                .append("Predicted Precipitation: ").append(predictedPrecipitation).append(" mm\n\n");
                    }

                    showForecastDialog(forecast.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(forecast.this, "Error parsing response", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(forecast.this, "Error fetching forecast", Toast.LENGTH_SHORT).show();
            }
        }

        private void showForecastDialog(String message) {
            AlertDialog.Builder builder = new AlertDialog.Builder(forecast.this);
            builder.setTitle("7-Day Weather Forecast");
            builder.setMessage(message);
            builder.setPositiveButton("OK", null);
            builder.show();
        }
    }
}
