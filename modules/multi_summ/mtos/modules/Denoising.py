# -*- coding : utf-8 -*-
# Denoising.py
# by thkim

# Code for implementing denosing mechanism related things

# Refer to readme.md about denoising mechanism
import torch
import torch.nn
import random

class Denoising():
  """
  Implements noise function used in denoisingmechanism for pytorch

  Args:
    noising_prob (:float): probability of noise

  """
  def __init__(self, noising_prob=0.1,  null_word_index=2, padding_index=1):
    self.noising_prob = noising_prob
    self.null_word_index = null_word_index
    self.padding_index = padding_index
    self.flag = 1
    
  def pad(self, tensor, length, pad_index=1):
    if isinstance(tensor, torch.autograd.Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var,
                Variable(pad_index * torch.ones(length - var.size(0), *var.size()[1:])).cuda().type_as(var)])
        else:
            return var
    else:
        #print("denoising line:35 tensor", tensor)
        #print("denoising line:36 pad", pad_index * torch.ones(length - tensor.size(0), *tensor.size()[1:]).cuda())
        if length > tensor.size(0):
            return torch.cat([tensor,
                                  pad_index * torch.ones(length - tensor.size(0), *tensor.size()[1:]).long().cuda()])
        else:
            return tensor      
    
  def seperate_noising(self, src, src_lengths):
    # src : len * batch * 1
    batch_size = src.size(1)
    
    noised_src_list = []
    altered_length_list = []
    noised_type_list = []
    
    for i in range(batch_size):
        #print("denoising line:49, src[:,i,:].data", src[:,i,:].data)
        noised_src, altered_length, noised_type = self.random_noise(src[:,i,:].data, src_lengths[i:i+1])
        #print("denoising line:51, noised_src", noised_src) # len -* 1
        #print("denoising line:52, noised_type", noised_type)
        noised_src_list.append(noised_src)
        altered_length_list.append(altered_length)
        noised_type_list.append(noised_type)
    altered_length_list = torch.cat(altered_length_list)
    noised_type_list = torch.LongTensor(noised_type_list).cuda()
    max_length = torch.max(altered_length_list)    
    
    noised_src_list = [ tensor if tensor.size(0) <= max_length else tensor[:max_length,:] for tensor in noised_src_list ]
    
    #print("denoising 60 max_length:", max_length)
    #print("denoising 61 pad indx:", self.padding_index)
    #print("denoising 62 noised_src_list[0]:", noised_src_list[0])
    try:
        noised_src = torch.stack([self.pad(tensor, max_length, self.padding_index) for tensor in noised_src_list ], 1)
    except:
        print("denoising noised_src_list 70:", noised_src_list)
        print("denoising noised_type_list 71:", noised_type_list)
        print("denoising altered_length_list 72:", altered_length_list)
        print("denoising max_length 73:", max_length)
        assert False
        input("denoising noised_src:71 err")
        
  def seperate_noising_all(self, src, src_lengths):
    # src : len * batch * 1
    batch_size = src.size(1)
    
    noised_src_list = []
    altered_length_list = []
    noised_type_list = []
    
    
    for i in range(batch_size):
        #print("denoising line:49, src[:,i,:].data", src[:,i,:].data)
        noised_src, altered_length, noised_type = self.random_noise_soft(src[:,i,:].data, src_lengths[i:i+1])
        #print("denoising line:51, noised_src", noised_src) # len -* 1
        #print("denoising line:52, noised_type", noised_type)
        noised_src_list.append(noised_src)
        altered_length_list.append(altered_length)
        noised_type_list.append(noised_type)
    altered_length_list = torch.cat(altered_length_list)
    noised_type_list = torch.FloatTensor(noised_type_list).cuda()
    max_length = torch.max(altered_length_list)    
    
    noised_src_list = [ tensor if tensor.size(0) <= max_length else tensor[:max_length,:] for tensor in noised_src_list ]
    
    #print("denoising 60 max_length:", max_length)
    #print("denoising 61 pad indx:", self.padding_index)
    #print("denoising 62 noised_src_list[0]:", noised_src_list[0])
    #input("denoising line:103")
    try:
        noised_src = torch.stack([self.pad(tensor, max_length, self.padding_index) for tensor in noised_src_list ], 1)
    except:
        print("denoising noised_src_list 70:", noised_src_list)
        print("denoising noised_type_list 71:", noised_type_list)
        print("denoising altered_length_list 72:", altered_length_list)
        print("denoising max_length 73:", max_length)
        assert False
        input("denoising noised_src:71 err")        
        
    
    #print(noised_src_list)
    #print(altered_length_list)
    
    #print("denoising noised_type_list:70", noised_type_list)
    #print("denoising noised_src:71", noised_src)
    
    return torch.autograd.Variable(noised_src), altered_length_list, noised_type_list
        
    
  def random_noise(self, src, src_lengths):
    """
    # src : len * batch * 1
    self.flag = 1 - self.flag
    if self.flag == 1: 
      noised_src, noised_src_lengths  = self.noise_drop_swap(src, src_lengths)
      return noised_src, noised_src_lengths, 0
    else: 
      noised_src, noised_src_lengths = self.noise_replace_swap(src, src_lengths)  
      return noised_src, noised_src_lengths, 1
    """
    
    if random.random() <= 0.5: 
      noised_src, noised_src_lengths, noise_cnt  = self.noise_drop_swap(src, src_lengths)
      return noised_src, noised_src_lengths, 0
    else: 
      noised_src, noised_src_lengths, noise_cnt = self.noise_replace_swap(src, src_lengths)  
      return noised_src, noised_src_lengths, 1
    
  # return 0.5
  def random_noise_soft(self, src, src_lengths):
    """
    # src : len * batch * 1
    self.flag = 1 - self.flag
    if self.flag == 1: 
      noised_src, noised_src_lengths  = self.noise_drop_swap(src, src_lengths)
      return noised_src, noised_src_lengths, 0
    else: 
      noised_src, noised_src_lengths = self.noise_replace_swap(src, src_lengths)  
      return noised_src, noised_src_lengths, 1
    """
    
    if random.random() <= 0.5: 
      noised_src, noised_src_lengths, noise_cnt  = self.noise_drop_swap(src, src_lengths)
      return noised_src, noised_src_lengths, 0 if noise_cnt > 1 else 0.5
    else: 
      noised_src, noised_src_lengths, noise_cnt = self.noise_replace_swap(src, src_lengths)  
      return noised_src, noised_src_lengths, 1 if noise_cnt > 1 else 0.5
    
    


  def noise_drop_swap(self, src, src_lengths):
    """
    Inject noise into src data and return noised data
    noise scheme is "drop+swap"

    Args:
      src (:obj Tensor (src_length * batchsize * 1)):
        original source sequence 
      src_lengths (:obj LongTensor) (batchsize)):
        sequence length of each elements in batch

    Returns:
      (:obj 'Float Tensor'(noised_src_length * batchsize * 1), :obj 'Long Tensor(batchsize)'
        * noised src sequence
        * nosied src lengths    
    
    """
    noise = [] # index of injecting noise
    p = self.noising_prob
    remainder = list(range(min(src_lengths), src.size(0)))
    noise_cnt = 0
    
    min_len = torch.min(src_lengths)
    
    if min_len <= 2:
      return src, src_lengths, noise_cnt

    # drop noise
    for i in range(min_len):
      if i - len(noise) >= min_len - 2:
        break
      if random.random() <= 1-p: 
        noise.append(i)
      else:
        noise_cnt += 1
                        
    # swap noise
    if len(noise) > 1:
      for i in range(len(noise)-1):
        if random.random() <= p:
          tmp = noise[i]
          noise[i] = noise[i+1]
          noise[i+1] = tmp
          i = i + 1
                        
    if len(noise) > 0:
      noised_len = len(noise)
      noise.extend(remainder)
      # assume training is processed on the gpu
    
      if isinstance(src, torch.autograd.Variable):
        selected_idx = torch.autograd.Variable(torch.LongTensor(noise)).cuda()
      else:
        selected_idx = torch.LongTensor(noise).cuda()
     
      noised_src = src.index_select(0, selected_idx ) # make new tensor
                    
      noised_src_lengths = src_lengths - (min_len-noised_len)
      
    else:
      noised_src = src
      noised_src_lengths = src_lengths

    return noised_src, noised_src_lengths, noise_cnt

  def noise_replace_swap(self, src, src_lengths):
    """
    Inject noise into src data and return noised data
    noise scheme is "drop+swap"

    Args:
      src (:obj Tensor (src_length * batchsize * 1)):
        original source sequence 
      src_lengths (:obj LongTensor) (batchsize)):
        sequence length of each elements in batch

    Returns:
      (:obj 'Float Tensor'(noised_src_length * batchsize * 1), :obj 'Long Tensor(batchsize)'
        * noised src sequence
        * nosied src lengths    
    
    """
    noise = list(range(min(src_lengths))) # index of injecting noise
    p = self.noising_prob
    remainder = list(range(min(src_lengths), src.size(0)))
    noise_cnt = 0

   
    # replace noise
    for i in range(min(src_lengths)):
      if random.random() <= p:
        #print("## denoise line:102 i",i)
        #print("## denoise line:102 src[i]",src[i])
        src[i] = self.null_word_index + src[i] * 0
        #print("## denoise line:102 src[i]",src[i])
        noise_cnt += 1
                        
    # swap noise
    
    for i in range(len(noise)-1):
      if random.random() <= p:
          tmp = noise[i]
          noise[i] = noise[i+1]
          noise[i+1] = tmp
          i = i + 1
                        
 #   if len(noise) > 0:
    noised_len = len(noise)
    noise.extend(remainder)
      # assume training is processed on the gpu
        
    if isinstance(src, torch.autograd.Variable):
      selected_idx = torch.autograd.Variable(torch.LongTensor(noise)).cuda()
    else:
      selected_idx = torch.LongTensor(noise).cuda()        
    noised_src = src.index_select(0, selected_idx ) # make new tensor
                    
    noised_src_lengths = src_lengths
#    else:
#      noised_src = src
#      noised_src_lengths = src_lengths

    return noised_src, noised_src_lengths, noise_cnt

  
  
    
