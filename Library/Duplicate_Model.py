###############################################################################
# This library provides utilities for duplicating two-sided and exchange 
# reactions in a SBML model. Examples of use are given in the notebook
# Duplicate_Model.ipynb
# Author: Leon Faure, leon.faure@inrae.fr
###############################################################################

# Libraries

import cobra
import numpy as np
import matplotlib.pyplot as plt
import requests

# Functions

def screen_out_in(model, io_dict, unsignificant_mols, problematic_reacs=[]):
    # The goal of this function is to screen all reactions of a model
    # recording if the reaction is encoding a change of compartment
    # for a metabolite, and in which way the reaction is encoded.
    # The purpose of this process is to obtain positive only fluxes.
    #
    # inputs:
    # - model: the original SBML model to be duplicated
    # - io_dict: dictionary mapping each suffix to be added to reactions to
    # the corresponding compartment-changing process
    # - unsignificant_mols: list of small molecules to be ignored in the 
    # screening process
    # - problematic_reacs: list of reactions to be ignored in the screening
    # process, by default it's empty and if some are detected, the function
    # is runned again (recursively) to omit these reactions
    #
    # output:
    # - reac_id_to_io_count_and_way: dictionary mapping each reaction to
    # a count of inflowing and outflowing compounds, and in which way the 
    # reaction happens. Used in the function duplicate_model()

    reac_id_to_io_count_and_way = {}

    # r = reversible, f = forward, b = backwards, o = other
    count_ways = {"r":0, "f":0, "b":0, "o":0}

    for reac in model.reactions:
        
        ub = reac.upper_bound
        lb = reac.lower_bound
        
        reversible = False
        only_forward = False
        only_backward = False
        other = False
        
        compartments_reactants = set()
        compartments_products = set()
        reactants = reac.reactants
        products = reac.products
        
        """
        1) For each reaction we retrieve the compartment of each reactant
        in the list "compartments_reactants" 
        and each product in the list "compartments_products"
        If no reactant/product, the compartment is "None".
        """
        
        if len(products) == 0:
            compartments_products.add(None)
        if len(reactants) == 0:
            # This should not be reached, when one-sided reaction its
            # the substrate side, always, for exchange and sinks
            compartments_reactants.add(None)
            print("Should not happen!")
        
        for reactant in reactants:
            if reactant.id in unsignificant_mols and \
            reac.id in problematic_reacs:
                continue
            comp = reactant.id.split('_')[-1]
            # reactions added in GalaxySynBioCad have a suffix MNXC3
            # for cytosol, special case treated
            if comp.endswith("MNXC3"):
                comp = "c"
            
            if comp in ["c", "p", "e", "m"]:
                compartments_reactants.add(comp)
                
        for product in products:
            if product.id in unsignificant_mols and \
            reac.id in problematic_reacs:
                continue
            comp = product.id.split('_')[-1]
            #reactions added by GalaxySBC have a suffix MNXC3 for cytosol
            if comp.endswith("MNXC3"):
                comp = "c"
                
            if comp in ["c", "p", "e", "m"]:
                compartments_products.add(comp)
                
        """
        2) We compare the list of compartments 
        and fill in the dictionary "io_count_dict"
        """

        if ub > 0 and lb < 0:
            reversible = True
            count_ways["r"] += 1
            way = "r"
        elif ub == 0 and lb < 0:
            only_backward = True
            count_ways["b"] += 1
            way = "b"
        elif lb == 0 and ub > 0:
            only_forward = True
            count_ways["f"] += 1
            way = "f"
        else:
            other = True
            count_ways["o"] += 1
            way = "o"

        if way == "r" or reac.id.startswith("EX_") or way == 'b': 
        # we don't retrieve forward reactions, 
        # they will not be duplicated or modified
            if compartments_reactants != compartments_products:
                
                # Make pairs of compartment transfers
                this_reac_pairs = []
                for comp_r in compartments_reactants:
                    for comp_p in compartments_products:
                        if comp_r == comp_p:
                            continue
                        this_reac_pairs.append((comp_r, comp_p))

                io_count_dict = {"_i": 0, "_o": 0}
                for pair in this_reac_pairs:
                    for key in io_dict.keys():
                        if pair in io_dict[key]:
                            io_count_dict[key] += 1
                
                # Was to screen problematic reactions, now useless since 
                # problematic reactions are already stored in a list
                if io_count_dict["_i"] != 0 and io_count_dict["_o"] != 0 \
                and reac.id not in problematic_reacs:
                    problematic_reacs.append(reac.id)
                    print("problematic reaction: ", reac.id, reac.reaction, 
                        io_count_dict)
                
                reac_id_to_io_count_and_way[reac.id] = [io_count_dict, way]

            else:
                # When the compartments are the same for reactants and 
                # products
                io_count_dict = {"_i": 0, "_o": 0}
                reac_id_to_io_count_and_way[reac.id] = [io_count_dict, way]

    if len(problematic_reacs) > 0:

        screen_out_in(model, io_dict, unsignificant_mols, problematic_reacs)

    print(count_ways)

    return reac_id_to_io_count_and_way


def duplicate_model(model, reac_id_to_io_count_and_way):
    # The goal of this function is to duplicate all reversible reactions
    # and all exchange reactions, excepted sink reactions. If a reaction
    # is coded backwards, we recode it forward.
    # The purpous of this process is to obtain positive only fluxes.
    #
    # inputs:
    # - model: the original model to be duplicated
    # - reac_id_to_io_count_and_way: dictionary mapping each reaction to
    # a count of inflowing and outflowing compounds, and in which way the 
    # reaction happens. Generated from the function screen_out_in()
    #
    # output:
    # - new_model: duplicated model

    new_model = model.copy()

    for key in reac_id_to_io_count_and_way.keys():
        i_or_o, way = reac_id_to_io_count_and_way[key]
        nbr_in = i_or_o["_i"]
        nbr_out = i_or_o["_o"]
        if nbr_in > nbr_out:
            suffix = "_i"
            rev_suffix = "_o"
        if nbr_out > nbr_in:
            suffix = "_o"
            rev_suffix = "_i"
        if (nbr_out == 0 and nbr_in == 0) or nbr_out == nbr_in:
            suffix = "_for"
            rev_suffix = "_rev"

        reac = new_model.reactions.get_by_id(key)
        
        if reac.id.startswith("DM_") or reac.id.endswith("sink") or \
        (not reac.id.startswith("EX_") and way == 'f') or way == 'o':
            # We do not duplicate the sink reactions,
            # either 'DM' for default sinks
            # or "sink" for GalaxySynBioCad-added sink
            # We do not duplicate irreversible forward internal reactions
            # We do not duplicate reactions with unclear way ("o")
            # Usually they are constrained to specific value (e.g. ATPM)
            continue

        # In this case (internal backward) we don't duplicate the reaction
        # we just encode it forward instead of backward
        if way == 'b' and not reac.id.startswith('EX_'):

            ori_name = reac.id
            new_name = reac.id + rev_suffix

            duplicated = cobra.Reaction(new_name)
            duplicated.bounds = (0, reversed_upper)

            new_model.add_reactions([duplicated])
            
            rev_metabolites = {}
            
            for key in reac.metabolites.keys():
                # print(key)
                rev_metabolites[key.id] = -reac.metabolites[key]
                
            duplicated.add_metabolites(rev_metabolites)

            new_model.remove_reactions([reac])

            new_model.reactions.get_by_id(new_name).id = ori_name 
            # we keep the same name as it was not duplicated
        
        # In all the other cases we duplicate
        else:
            # Following procedure is to DUPLICATE reactions 
            # (for reversible reactions, AND exchange forward or backward)
            if way == 'r':
                reversed_upper = - reac.lower_bound
                normal_upper = reac.upper_bound
            if way == 'f':
                reversed_upper = reac.upper_bound
                normal_upper = reac.upper_bound
            if way == 'b':
                reversed_upper = - reac.lower_bound
                normal_upper = - reac.lower_bound

            reac.bounds = (0, normal_upper) 
            # Always put the bounds at (0, 1000), even for the duplicated
            
            reactants = reac.reactants
            products = reac.products

            duplicated = cobra.Reaction(reac.id + rev_suffix)
            duplicated.bounds = (0, reversed_upper)

            new_model.add_reactions([duplicated])
            
            rev_metabolites = {}
            
            for key in reac.metabolites.keys():
                # print(key)
                rev_metabolites[key.id] = -reac.metabolites[key]
                
            duplicated.add_metabolites(rev_metabolites)
            
            reac.id += suffix

    print("The default model had " + str(len(model.reactions)) + 
        " reactions and the duplicated-reactions model has " + 
        str(len(new_model.reactions)) + " reactions.")
    return new_model


def correct_duplicated_med(default_med, duplicated_med):
    # The goal of this function is to set the right upper bound values
    # for exchange reactions so that the medium object of the duplicated
    # model is coherent with the original model
    # 
    # inputs:
    # - default_med: medium object (generated by cobra) of the original model
    # - duplicated_med: medium object of the duplicated model 
    #
    # output:
    # - duplicated_med: corrected duplicated_med

    for duplicated in duplicated_med.keys():
        # print(duplicated_med[duplicated])
        duplicated_med[duplicated] = 1e-300
        for default in default_med.keys():
            if duplicated.startswith(default):
                duplicated_med[duplicated] = default_med[default]
    return duplicated_med


def randomize_default_medium(medium, seed):
    # function to transform a medium's values with random drawings
    # between 0 and the default value, for varying the upper bounds
    # 
    # inputs:
    # - medium object to be changed
    # - seed for changing random drawings

    np.random.seed(seed)

    for key in medium.keys():
        random_val_up = np.random.uniform(0, medium[key])
        medium[key] = random_val_up

    return medium

def copy_from_default(def_med, dup_med):
    # function to get the same medium for the duplicated model
    # as for the original one
    # 
    # inputs:
    # - def_med: medium to be copied into dup_med
    # - dup_med: medium to be changed

    for key in def_med.keys():
        for key_dup in dup_med.keys():
            if key_dup.startswith(key):
                dup_med[key_dup] = def_med[key]
                break
    return dup_med

def change_medium(model, duplicated_model, seed):
    # function to randomize a medium object then optimize two models
    # (one with duplicated reactions) with the same medium
    # and retrieve their solutions
    #
    # inputs:
    # - model: original model
    # - duplicated_model: duplicated model
    # - seed: for changing random drawings

    with model as def_mod:
        with duplicated_model as dup_mod:
            
            def_med = def_mod.medium
            changed_def_med = randomize_default_medium(def_med, seed)
            def_mod.medium = changed_def_med
            dup_med = dup_mod.medium
            changed_dup_med = copy_from_default(changed_def_med, dup_med)
            dup_mod.medium = changed_dup_med

            new_in_dup = set([x[:-2] if x.endswith('_i') else x for x in
                dup_med.keys()]) - set(def_med.keys())
            # used for checking new elements of medium in duplicated medium

            s = def_mod.optimize()
            s_dup = dup_mod.optimize()
            s_val = None
            s_val_dup = None
            if s.objective_value > 0 or s.status != 'infeasible':
                s_val = s.objective_value
            if s_dup.objective_value > 0 and s_dup.status != 'infeasible':
                s_val_dup = s_dup.objective_value

    return s_val, s_val_dup