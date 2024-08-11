package com.example.climacare;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;

public class CallPageActivity extends BaseActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.call_layout);

        // Set up the ImageView click listeners
        ImageView redCrossLogo = findViewById(R.id.redCrossLogo);
        ImageView policeLogo = findViewById(R.id.policeLogo);
        ImageView amnestyInternationalLogo = findViewById(R.id.amnestyInternationalLogo);
        ImageView johnLogo = findViewById(R.id.john);
        ImageView unicefLogo = findViewById(R.id.Unicef);
        ImageView whoLogo = findViewById(R.id.WHO);

        redCrossLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:+254715820219");
            }
        });

        policeLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:999");
            }
        });

        amnestyInternationalLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:+254733444555");
            }
        });

        johnLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:+254711697379");
            }
        });

        unicefLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:+254722515680");
            }
        });

        whoLogo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                makePhoneCall("tel:+254207276000");
            }
        });
    }

    private void makePhoneCall(String phoneNumber) {
        Intent intent = new Intent(Intent.ACTION_DIAL);
        intent.setData(Uri.parse(phoneNumber));
        startActivity(intent);
    }
}
