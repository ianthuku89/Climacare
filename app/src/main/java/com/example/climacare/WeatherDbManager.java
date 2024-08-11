package com.example.climacare;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class WeatherDbManager {

    private static final String DATABASE_NAME = "weather_predictions.db";
    private static final int DATABASE_VERSION = 1;
    private static final String TABLE_NAME = "predictions";
    private static final String COLUMN_ID = "_id";
    private static final String COLUMN_LOCATION = "location";
    private static final String COLUMN_PREDICTION = "prediction";
    private static final String COLUMN_PREDICTED_TEMPERATURE = "predicted_temperature";
    private static final String COLUMN_PREDICTED_PRECIPITATION = "predicted_precipitation";

    private final Context context;
    private DatabaseHelper DBHelper;
    private SQLiteDatabase db;

    public WeatherDbManager(Context ctx) {
        this.context = ctx;
        DBHelper = new DatabaseHelper(context);
    }

    public Cursor getAllPredictions() {
        return db.query(TABLE_NAME, null, null, null, null, null, null);
    }

    public int clearAllPredictions() {
        return db.delete(TABLE_NAME, null, null);
    }

    private static class DatabaseHelper extends SQLiteOpenHelper {

        DatabaseHelper(Context context) {
            super(context, DATABASE_NAME, null, DATABASE_VERSION);
        }

        @Override
        public void onCreate(SQLiteDatabase db) {
            String CREATE_TABLE = "CREATE TABLE " + TABLE_NAME + " ("
                    + COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                    + COLUMN_LOCATION + " TEXT NOT NULL, "
                    + COLUMN_PREDICTION + " TEXT NOT NULL, "
                    + COLUMN_PREDICTED_TEMPERATURE + " REAL NOT NULL, "
                    + COLUMN_PREDICTED_PRECIPITATION + " REAL NOT NULL);";
            db.execSQL(CREATE_TABLE);
        }

        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            db.execSQL("DROP TABLE IF EXISTS " + TABLE_NAME);
            onCreate(db);
        }
    }

    public WeatherDbManager open() throws SQLException {
        db = DBHelper.getWritableDatabase();
        return this;
    }

    public void close() {
        DBHelper.close();
    }

    public long addPrediction(String location, String prediction, double predictedTemperature, double predictedPrecipitation) {
        ContentValues initialValues = new ContentValues();
        initialValues.put(COLUMN_LOCATION, location);
        initialValues.put(COLUMN_PREDICTION, prediction);
        initialValues.put(COLUMN_PREDICTED_TEMPERATURE, predictedTemperature);
        initialValues.put(COLUMN_PREDICTED_PRECIPITATION, predictedPrecipitation);

        return db.insert(TABLE_NAME, null, initialValues);
    }
}
