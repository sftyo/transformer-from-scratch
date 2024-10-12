# transformer from scratch!
The aim of this project is to understand the *Attention* mechanism from this infamous [paper](https://arxiv.org/abs/1706.03762), especially the decoder block (taken from GPT2 architecture). The goal is to predict the next word given the context of the word. For example, given the sentence "marry want a little lamb", we want to know what comes after. In this project the GPT2 tokenizer will be used (BPE), although it is also possible to create your own tokenizer.

*Note: this is an on going repo, meaning new things will be added, such as RoPE embedding, GQA, etc..*

# dataset
In this project we are going to use text data from Shakespeare's novel. With the objective to produce a shakespearean texts.

# results
Since I am running this locally in my laptop, we will be using a small model, with parameters `context_length: 64, n_head: 4, n_layers: 4, d_model: 128 and batch_size: 64`, with approximately 13M parameters.

Here are the **training/val** loss result plot:
![Figure_1](https://github.com/user-attachments/assets/339f574a-6fba-4ddf-b90d-3cc2f38f2cb5)

As we can see that the model is overfitting, even though we use 10> epochs, this is due to the small dataset that was used. We can try to use a larger dataset, but I don't think my laptop could handle it, and as the goal of this project is to understand the *Attention* mechanism, i think its already sufficient enough.

By using "Leaving this place is something" as our initial output, we get the output of the text generating as:
"Leaving this place is something .


 The.H had me's at I had: lift ond persu I by the half's in medium of andd sketch sketch can-- Rick? of I see here--burn's--I's--
I?_ rest


 The head; how saw it happened glanced it, at I was with surprise; happened me me.


I his was-- . work to exasper the last-- Toham it--I; the of but was only, the cigars it.

 That was for had .'s to think to . when had he had, his_ pictures_ had, it.I--ir that it was here I, still me he he historyte me said first satisfaction them."      

 The foundations.
burn what,; wasa him from--

" . had it out and.

 uncertain, my him are, the_., the, and it hadThere, and to--"

 As you can see, since the loss is not very good, we have a lot of non-sensical word and a lot of typos.
