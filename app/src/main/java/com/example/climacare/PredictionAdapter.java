package com.example.climacare;

import android.content.Context;
import android.database.Cursor;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CursorAdapter;
import android.widget.TextView;

import java.text.DecimalFormat;

public class PredictionAdapter extends CursorAdapter {

    // Create a DecimalFormat instance to format numbers to two decimal places
    private DecimalFormat decimalFormat = new DecimalFormat("0.00");

    public PredictionAdapter(Context context, Cursor cursor) {
        super(context, cursor, 0);
    }

    @Override
    public View newView(Context context, Cursor cursor, ViewGroup parent) {
        return LayoutInflater.from(context).inflate(R.layout.list_item_prediction, parent, false);
    }

    @Override
    public void bindView(View view, Context context, Cursor cursor) {
        TextView predictionEntryTextView = view.findViewById(R.id.predictionEntryTextView);
        String location = cursor.getString(cursor.getColumnIndexOrThrow("location"));
        String prediction = cursor.getString(cursor.getColumnIndexOrThrow("prediction"));
        double predictedTemperature = cursor.getDouble(cursor.getColumnIndexOrThrow("predicted_temperature"));
        double predictedPrecipitation = cursor.getDouble(cursor.getColumnIndexOrThrow("predicted_precipitation"));

        // Format the temperature and precipitation values to two decimal places
        String formattedTemperature = decimalFormat.format(predictedTemperature);
        String formattedPrecipitation = decimalFormat.format(predictedPrecipitation);

        // Construct the prediction entry text
        String predictionEntry = "Location: " + location +
                "\nPrediction: " + prediction +
                "\nPredicted Temperature : " + formattedTemperature + "°C" +
                "\nPredicted Precipitation: " + formattedPrecipitation + " mm";

        predictionEntryTextView.setText(predictionEntry);
    }

    public void clear() {
    }
}
