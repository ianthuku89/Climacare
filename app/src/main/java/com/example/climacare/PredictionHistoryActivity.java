package com.example.climacare;

import android.database.Cursor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class PredictionHistoryActivity extends BaseActivity {

    private WeatherDbManager dbManager;
    private PredictionAdapter adapter;
    private ListView predictionHistoryListView;
    private Button clearButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.prediction_history_layout);

        // Initialize database manager
        dbManager = new WeatherDbManager(this);
        dbManager.open();

        predictionHistoryListView = findViewById(R.id.predictionHistoryListView);
        clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clearPredictionHistory();
            }
        });

        showPredictionHistory();
    }

    // Method to retrieve prediction history from database
    private void showPredictionHistory() {
        Cursor cursor = dbManager.getAllPredictions();
        if (adapter == null) {
            adapter = new PredictionAdapter(this, cursor);
            predictionHistoryListView.setAdapter(adapter);
        } else {
            adapter.changeCursor(cursor);
            adapter.notifyDataSetChanged();
        }
    }

    // Method to clear prediction history from database
    private void clearPredictionHistory() {
        int rowsDeleted = dbManager.clearAllPredictions();
        if (rowsDeleted > 0) {
            Toast.makeText(this, "Prediction history cleared", Toast.LENGTH_SHORT).show();
            // Refresh the list view
            showPredictionHistory();
        } else {
            Toast.makeText(this, "No predictions to clear", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Close the database when the activity is destroyed
        dbManager.close();
    }
}
