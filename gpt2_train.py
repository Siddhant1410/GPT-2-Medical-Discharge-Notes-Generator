from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    return train_dataset,test_dataset,data_collator

def train(model_path,training_args,train_dataset,test_dataset,data_collator):
    model = GPT2LMHeadModel.from_pretrained(model_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model()

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    train_dataset,test_dataset,data_collator = load_dataset('train_1.txt','test_1.txt',tokenizer)

    training_args = TrainingArguments(
        output_dir="gpt2_finetuned", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=3, # number of training epochs
        evaluation_strategy='steps', # Validation accuracy to print after every eval_steps
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        eval_steps = 200, # Number of update steps between two evaluations.
        save_steps=400, # after # steps model is saved
        warmup_steps=100,# number of warmup steps for learning rate scheduler
        logging_steps=1,
        )

    train('gpt2', training_args,train_dataset,test_dataset,data_collator)

if __name__ == '__main__':
    main()

