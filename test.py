from transformers import GPT2LMHeadModel, GPT2Tokenizer
from icd9cms.icd9 import search
icd9_code = []

def generate_text(input_text, model_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

    inputs = tokenizer.encode(input_text, return_tensors='pt')

    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.7, no_repeat_ngram_size=2) #tweakable parameters
    
#    for i in range(5):
#        print(tokenizer.decode(outputs[i], skip_special_tokens=True))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    


def main():
    admn_date = input("Enter Patient's Admission date: ")
    sex = input("Sex: Male / Female?: ")
    
    diag_icd = input("Patient diagnosis icd9 code: ")
    diagnosis = search(diag_icd)
    diagnosis = str(diagnosis).replace(str(diag_icd), "")
    diagnosis = diagnosis.replace(":", ", ") + "."
    print("Pateint Summary: ")
    input_text = "Patient was Admitted on Date: "+ admn_date +". " + "Patient is a " + sex+" " + "and has been Diagnosed with: " + diagnosis + ". " + "The patient is advised to"
    model_path = "./gpt2_medium_finetuned"  # path to the fine-tuned model
    output_text = generate_text(input_text, model_path)

    print(output_text)

if __name__ == '__main__':
    main()
