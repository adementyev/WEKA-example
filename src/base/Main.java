package base;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;


public class Main {

	public static void main(String[] args) throws Exception{
		
		// Declare two numeric attributes
        Attribute Attribute1 = new Attribute("height");
        Attribute Attribute2 = new Attribute("weight");
        
        // Declare the location attribute
        FastVector Location = new FastVector(3);
        Location.addElement("australia");
        Location.addElement("africa");
        Location.addElement("america");
        Attribute Attribute3 = new Attribute("location", Location);
        
        //Declare the class attribute
        FastVector Classes = new FastVector(2);
        Classes.addElement("cat");
        Classes.addElement("elephant");
        Attribute ClassAttribute = new Attribute("Class", Classes);
        
        
        // Make the feature vector
        FastVector fvWekaAttributes = new FastVector(4);
        fvWekaAttributes.addElement(Attribute1);    
        fvWekaAttributes.addElement(Attribute2);    
        fvWekaAttributes.addElement(Attribute3);    
        fvWekaAttributes.addElement(ClassAttribute);
        
        // Create an empty training set
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);   
        
        // Set class index
        isTrainingSet.setClassIndex(3);
        
        // Create a training instance
        Instance iExample = new DenseInstance(4);
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 0.3);      
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 6);      
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "america");
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "cat");
        
        // add the instance
        isTrainingSet.add(iExample);
        
        // Create a second training instance
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 10);      
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 500);      
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "africa");
        iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "elephant");
        
        // add the second training instance
        isTrainingSet.add(iExample);
        
        System.out.println("The training data: " + isTrainingSet);
        
        //Build the classifier
        Classifier model = (Classifier)new SMO();   
        model.buildClassifier(isTrainingSet);
        
        //Create test data
        Instances evalSet = new Instances("Rel", fvWekaAttributes, 10);   
        evalSet.setClassIndex(3);
        Instance testInstance = new DenseInstance(4);
        testInstance.setValue((Attribute)fvWekaAttributes.elementAt(0), 5);      
        testInstance.setValue((Attribute)fvWekaAttributes.elementAt(1), 500);      
        testInstance.setValue((Attribute)fvWekaAttributes.elementAt(2), "africa");      
        evalSet.add(testInstance);
        
        System.out.println("The evaluation data: " + evalSet.instance(0));
        
        //Classify new Instance
        double ClassLabel = 100;
        ClassLabel = model.classifyInstance(evalSet.instance(0)); 
        System.out.println("Classified: " + ClassLabel);
	}

}
