package com.example.climacare;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class WeatherDbHelper extends SQLiteOpenHelper {

    private static final String DATABASE_NAME = "weather_predictions.db";
    private static final int DATABASE_VERSION = 1;

    public WeatherDbHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        // Create table to store weather predictions
        db.execSQL("CREATE TABLE predictions (" +
                "_id INTEGER PRIMARY KEY AUTOINCREMENT," +
                "location TEXT," +
                "prediction TEXT," +
                "predicted_temperature REAL," +
                "predicted_precipitation REAL)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // Upgrade logic, if needed
    }
}
