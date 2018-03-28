package com.example.dell.myapplication;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import java.text.AttributedCharacterIterator;
import java.util.ArrayList;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        NaiveBayes bayes = null;
        try {
            bayes= (NaiveBayes) weka.core.SerializationHelper.read(getAssets().open("naive.model"));
          //  Toast.makeText(this, "Success", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
        }
        final Attribute  attributeAge=new Attribute("age");
        final  Attribute attributeMass=new Attribute("mass");
        final ArrayList<String>classes=new ArrayList();
        classes.add("tested_positive");
        classes.add("tested_negative");
        ArrayList <Attribute> attributeList=new ArrayList<>();
        Attribute classAtrribute=new Attribute("class",classes);
        attributeList.add(attributeAge);
        attributeList.add(attributeMass);
        attributeList.add(classAtrribute);
        Instances dataUnpredicted = new Instances("TestInstances",
                attributeList, 1);

        dataUnpredicted.setClassIndex(dataUnpredicted.numAttributes()-1);
        DenseInstance instance=new DenseInstance(dataUnpredicted.numAttributes());
        instance.setValue(attributeAge,30);
        instance.setValue(attributeMass,80);
        DenseInstance newInstance = instance;
        // reference to dataset
        newInstance.setDataset(dataUnpredicted);
        try {
            Double result=bayes.classifyInstance(instance);
            String output=classes.get(new Double(result).intValue());
            Log.d("OUTPUT",output.toString());
            Toast.makeText(this, output, Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e("error",e.getMessage());
        }


    }
}
