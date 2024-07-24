import json 
import numpy as np


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data



class DataType:
    TRAINING=1
    EVALUATION=2
    TEST=3

class ArcKaggleDataLoader():

    

    def __init__(self,path):
        self.base_path = path

        self.training_challenges   = load_json(self.base_path +'arc-agi_training_challenges.json')
        self.training_solutions    = load_json(self.base_path +'arc-agi_training_solutions.json')

        self.evaluation_challenges = load_json(self.base_path +'arc-agi_evaluation_challenges.json')
        self.evaluation_solutions  = load_json(self.base_path +'arc-agi_evaluation_solutions.json')

        self.test_challenges       = load_json(self.base_path +'arc-agi_test_challenges.json')

        self.training_keys = list(self.training_challenges.keys())
        self.evaluation_keys = list(self.evaluation_challenges.keys())
        self.test_keys = list(self.test_challenges.keys())

        self.data_keys = {
            DataType.TRAINING:self.training_keys,
            DataType.EVALUATION:self.evaluation_keys,
            DataType.TEST:self.test_keys
        }

        self.challenges = {
            DataType.TRAINING:self.training_challenges,
            DataType.EVALUATION:self.evaluation_challenges,
            DataType.TEST:self.test_challenges
        }

        self.solutions = {
            DataType.TRAINING:self.training_solutions,
            DataType.EVALUATION:self.evaluation_solutions,
            DataType.TEST:None
        }

    def get_challenge(self,idx,type=DataType.TRAINING):
        assert idx>=0 and idx < len(self.data_keys[type])
        key = self.data_keys[type][idx]

        trainExamples = self.challenges[type][key]["train"]
        trainExamplesInput = [np.array(example["input"]) for example in trainExamples]
        trainExamplesOutput = [np.array(example["output"]) for example in trainExamples]

        testInput = np.array(self.challenges[type][key]["test"][0]["input"])
        testOutput = np.array(self.solutions[type][key][0])

        return ArcProblem(trainExamplesInput,trainExamplesOutput,testInput,testOutput,key)



    def create_full_text_training_dataset(self,type=DataType.TRAINING,transposed=True,rotated=True,transpose_rotated=False):
        AllTexts = []
        keys = self.data_keys[type]
        for i in range(len(keys)):
            problem = self.get_challenge(i,type)
            training_example = problem.get_problem_as_llm_string()
            AllTexts.append(training_example)
            

            other_problems=[]
            if transposed:
                other_problems.append(problem.create_transposed_problem())
            if rotated:
                other_problems.extend(problem.create_rotated_problem())
            if transpose_rotated:
                other_problems.extend(problem.create_transposed_problem().create_rotated_problem())

            for p in other_problems:
                training_example= p.get_problem_as_llm_string()
                AllTexts.append(training_example)


        return AllTexts
    
    def create_full_text_evaluation_dataset(self,type=DataType.EVALUATION,transposed=True,rotated=False,transpose_rotated=False):
        AllTexts = []
        solutions = []
        keys = self.data_keys[type]
        problems = []
        for i in range(len(keys)):
            
            problem = self.get_challenge(i,type)
            
            
            training_example = problem.get_problem_as_llm_string()
            x,y = problem.get_challenge_as_string()
            training_example+= "[EXAMPLE START] \n"+x
            
            AllTexts.append(training_example)
            problems.append(problem)
            solutions.append(y)


            other_problems=[]
            if transposed:
                other_problems.append(problem.create_transposed_problem())
            if rotated:
                other_problems.extend(problem.create_rotated_problem())
            if transpose_rotated:
                other_problems.extend(problem.create_transposed_problem().create_rotated_problem())

            for p in other_problems:
                training_example = p.get_problem_as_llm_string()
                x,y = p.get_challenge_as_string()
                training_example+= "[EXAMPLE START] \n"+x
                
                AllTexts.append(training_example)
                problems.append(problem)
                solutions.append(y)

        return AllTexts,solutions,problems
    

class ArcProblem():

    def __init__(self, train_inputs, train_outputs, challenge_input, challenge_output=None, challenge_code="",single_output=True, remap_colors=True):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        assert len(train_inputs) == len(train_outputs)
        self.challenge_input = challenge_input
        self.challenge_output = challenge_output
        if single_output:
            if self.challenge_output is not None and len(self.challenge_output.shape)==3:
                self.challenge_output=self.challenge_output[0]

        self.num_colors = 10
        self.challenge_code = challenge_code
        self.mapping = {i:i for i in range(self.num_colors)}
        self.rev_mapping = {i:i for i in range(self.num_colors)}
        if remap_colors:
            self.remap_colors_()

    def __len__(self):
        return len(self.train_inputs)

    def get_number_examples(self):
        return len(self)
    
    @staticmethod
    def matrix_to_integer_string_list(mat):
        "Slow, make quicker if necessary #TODO"
        s=[]
        for row in mat:
            temp="["
            for val in row:
                temp+=str(int(val))+","

            temp = temp[:-1] + "]"
            s.append(temp)
        return s

    def get_train_example_raw(self, idx):
        return self.train_inputs[idx], self.train_outputs[idx]
    

    def get_train_example_as_string(self, idx):
        input_value,output_value = self.get_train_example_raw(idx) 

        textIn = "\n".join(ArcProblem.matrix_to_integer_string_list(input_value))
        textOut = "\n".join(ArcProblem.matrix_to_integer_string_list(output_value))

        return "INPUT: \n"+ textIn + "\n OUTPUT: \n" + textOut
        
    def get_test_example_as_string(self):
        textIn = "\n".join(ArcProblem.matrix_to_integer_string_list(self.challenge_input))
        textOut = "\n".join(ArcProblem.matrix_to_integer_string_list(self.challenge_output))

        return "Input: \n"+ textIn + "\n OUTPUT: \n", textOut
    
    def get_problem_as_llm_string(self):
        text = ""
        for i in range(len(self)):
            text+=  "[EXAMPLE START] \n"+self.get_train_example_as_string(i)+"\n [EXAMPLE END] \n"
        return text

    def get_challenge_as_string(self):
        textIn = "\n".join(ArcProblem.matrix_to_integer_string_list(self.challenge_input))
        if self.challenge_output is None:
            return textIn,None
        textOut = "\n".join(ArcProblem.matrix_to_integer_string_list(self.challenge_output))

        return textIn,textOut

    def remap_colors_(self):
        """
        Remap colors, such that the most common color over all training examples is mapped to 0, the second most common to 1, etc.
        """
        from collections import Counter

        colors = Counter(range(10))
        for i in range(len(self)):
            colors.update(self.train_inputs[i].flatten())
            colors.update(self.train_outputs[i].flatten())
        colors = colors.most_common()

        mapping = {color: i for i, (color, _) in enumerate(colors)}
        self.mapping = mapping
        self.rev_mapping ={self.mapping[key]:key for key in self.mapping}
        for i in range(len(self)):
            self.train_inputs[i] = np.vectorize(mapping.get)(self.train_inputs[i])
            self.train_outputs[i] = np.vectorize(mapping.get)(self.train_outputs[i])
        self.challenge_input = np.vectorize(mapping.get)(self.challenge_input)
        if self.challenge_output is not None:
            self.challenge_output = np.vectorize(mapping.get)(self.challenge_output)

    def reverse_color_map(self,matrix):
        return np.vectorize(self.rev_mapping.get)(matrix)


    def create_transposed_problem(self):
        """
        Add transposed examples to the training set.
        """
        num_examples = len(self)
        train_inputs = []
        train_outputs = []
        for i in range(num_examples):
            train_inputs.append(np.transpose(self.train_inputs[i]))
            train_outputs.append(np.transpose(self.train_outputs[i]))

        challenge_input = np.transpose(self.challenge_input)
        if self.challenge_output is not None:
            challenge_output = np.transpose(self.challenge_output)
        else:
            challenge_output=None

        return ArcProblem(train_inputs,train_outputs,challenge_input,challenge_output,self.challenge_code+"_T")

    def create_rotated_problem(self, rotations=[1,2,3]):
        """
        Return rotated problem
        """
        num_examples = len(self)
        new_problems = []

        for k in rotations:

            train_inputs = []
            train_outputs = []
            for i in range(num_examples):
                train_inputs.append(np.rot90(self.train_inputs[i], k))
                train_outputs.append(np.rot90(self.train_outputs[i], k))
            
           
            challenge_input = np.rot90(self.challenge_input,k)
            if self.challenge_output is not None:
                challenge_output = np.rot90(self.challenge_output,k)
            else:
                challenge_output = None
            new_problems.append(ArcProblem(train_inputs,train_outputs,challenge_input,challenge_output,self.challenge_code+"_R"+str(k)))

        return new_problems