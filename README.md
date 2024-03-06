# PLEXOS API to set up and run net zero model
Chris Matthew

This project uses the PLEXOS API to incrementally add PLEXOS objects by the numbered scripts 
then runs the final model and collects the results. The scripts can be run in sequence to 
troubleshoot aspects of each model, where each one adds elements to the blank file included
in that folder.
Alternatively, it can be run using the 5.plexos_master_run file, which
will compile the model using the other scripts to add the required elements. This script will
also run the model for a given list of scenarios, then compile the results in one folder.
