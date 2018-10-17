//
//  ViewController.swift
//  Iris-classifier
//
//  Created by meraj on 2018/10/17.
//  Copyright © 2018 meraj. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {
    var model: Iris?
    var iris_data = [5.1, 3.5, 1.4, 0.2]
   
    override func viewDidLoad() {
        super.viewDidLoad()
    
        model = Iris()
    
        guard let input = try? MLMultiArray(shape:[1,4], dataType: MLMultiArrayDataType.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        
        for (index, element) in iris_data.enumerated() {
            input[index] = NSNumber(floatLiteral: element)
        }
        
        guard let prediction = try? model?.prediction(input: input) else {
            return
        }
        
        print("Predicted class: \(prediction!.classLabel)")
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
}

