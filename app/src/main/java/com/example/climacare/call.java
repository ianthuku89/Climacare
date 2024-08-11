package com.example.climacare;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class call extends BaseActivity {

    private static final int REQUEST_CALL_PERMISSION = 1;
    private String phoneNumber;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.call_layout);

        ImageView redCross = findViewById(R.id.redCrossLogo);
        ImageView police = findViewById(R.id.policeLogo);
        ImageView amnestyInternational = findViewById(R.id.amnestyInternationalLogo);
        ImageView john = findViewById(R.id.john);
        ImageView unicef = findViewById(R.id.Unicef);
        ImageView who = findViewById(R.id.WHO);

        redCross.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "+254715820219";
                makePhoneCall();
            }
        });

        police.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "999";
                makePhoneCall();
            }
        });

        amnestyInternational.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "+254733444555";
                makePhoneCall();
            }
        });

        john.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "+254711697379";
                makePhoneCall();
            }
        });

        unicef.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "+254722515680";
                makePhoneCall();
            }
        });

        who.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                phoneNumber = "+254207276000";
                makePhoneCall();
            }
        });
    }

    private void makePhoneCall() {
        if (ContextCompat.checkSelfPermission(call.this,
                Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(call.this,
                    new String[]{Manifest.permission.CALL_PHONE}, REQUEST_CALL_PERMISSION);
        } else {
            startCall(phoneNumber);
        }
    }

    private void startCall(String phoneNumber) {
        Intent callIntent = new Intent(Intent.ACTION_CALL);
        callIntent.setData(Uri.parse("tel:" + phoneNumber));
        startActivity(callIntent);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CALL_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                makePhoneCall();
            } else {
                // Permission denied, show a message to the user
                // For example: Toast.makeText(this, "Permission DENIED", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
